# main.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator


# =========================
# Tipos
# =========================

Side = Literal["BUY", "SELL"]
OrderStatus = Literal["FILLED"]


# =========================
# Entrada (ordem Arena/Nelogica)
# =========================

class ArenaOrderIn(BaseModel):
    id: str
    code: str
    side: str          # "0" ou "1" (ou pode vir BUY/SELL no futuro)
    price: str
    active: bool

    # compatibilidade:
    # - antigo: status="1" (executada)
    # - novo: status=0 e statusCalculate=1 (executada)
    status: Optional[str] = None
    statusCalculate: Optional[int] = None

    tradeId: Optional[str] = None
    dateTime: datetime
    personId: Optional[str] = None
    quantity: str
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    groupOrderId: Optional[str] = None
    tournamentId: Optional[str] = None
    tournamentPersonId: Optional[str] = None
    token: Optional[str] = None
    operationId: Optional[str] = None

    @field_validator("dateTime")
    @classmethod
    def ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class OrdersPayload(BaseModel):
    orders: List[ArenaOrderIn]


class BodyPayload(BaseModel):
    body: List[ArenaOrderIn]


# =========================
# Canonical interno
# =========================

class OrderIn(BaseModel):
    order_id: str
    timestamp: datetime
    symbol: str
    side: Side
    qty: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    status: OrderStatus = "FILLED"

    @field_validator("timestamp")
    @classmethod
    def ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


# =========================
# Saída
# =========================

class CleanOrder(BaseModel):
    order_id: str
    timestamp: datetime
    symbol: str
    symbol_prefix: str
    side: Side
    qty: float
    price: float
    status: OrderStatus


class OperationOut(BaseModel):
    symbol_prefix: str
    symbol: str
    direction: Literal["LONG", "SHORT"]
    qty: float
    open_time: datetime
    close_time: datetime
    duration_sec: float
    avg_entry_price: float
    avg_exit_price: float
    pnl_points: float
    pnl_money: Optional[float] = None
    trades_count: int


class OpenPositionOut(BaseModel):
    symbol_prefix: str
    symbol: str
    direction: Literal["LONG", "SHORT"]
    qty: float
    avg_entry_price: float
    open_time: datetime


class Module0Response(BaseModel):
    request_id: str
    meta: Dict[str, Any]
    orders_clean: List[CleanOrder]
    operations: List[OperationOut]
    open_positions: List[OpenPositionOut]


# =========================
# Helpers
# =========================

def symbol_prefix(symbol: str) -> str:
    return symbol[:3].upper().strip()


# mantém seu padrão: "0"=BUY e "1"=SELL
SIDE_MAP = {"0": "BUY", "1": "SELL", "BUY": "BUY", "SELL": "SELL"}

# regras de executada:
# - se status == "1" => executada (antigo)
# - se statusCalculate == 1 => executada (novo)
def is_executed(a: ArenaOrderIn) -> bool:
    if a.status is not None and str(a.status).strip() == "1":
        return True
    if a.statusCalculate is not None and int(a.statusCalculate) == 1:
        return True
    return False


def arena_to_orderin(a: ArenaOrderIn) -> Optional[OrderIn]:
    if not a.active:
        return None
    if not is_executed(a):
        return None

    side = SIDE_MAP.get(str(a.side).strip().upper())
    if not side:
        return None

    try:
        qty = float(a.quantity)
        price = float(a.price)
    except Exception:
        return None

    if qty <= 0 or price <= 0:
        return None

    return OrderIn(
        order_id=a.id,
        timestamp=a.dateTime,
        symbol=a.code,
        side=side,  # type: ignore
        qty=qty,
        price=price,
        status="FILLED",
    )


def signed_qty(side: Side, qty: float) -> float:
    return qty if side == "BUY" else -qty


def close_qty_portion(pos_qty: float, delta_qty: float) -> float:
    return min(abs(pos_qty), abs(delta_qty))


# =========================
# Core – gera operações por prefixo
# =========================

class _BuildResult(BaseModel):
    operations: List[OperationOut]
    open_position: Optional[OpenPositionOut] = None


def build_operations_for_prefix(orders: List[OrderIn], point_value: Optional[float]) -> _BuildResult:
    ops: List[OperationOut] = []

    pos_qty = 0.0
    avg_entry = 0.0
    op_open_time: Optional[datetime] = None
    op_direction: Optional[str] = None
    open_symbol: Optional[str] = None
    pref: Optional[str] = None

    exit_notional = 0.0
    exit_qty_accum = 0.0
    trades_count = 0
    op_pnl_points = 0.0

    def finalize_op(close_time: datetime):
        nonlocal op_open_time, op_direction, open_symbol, pref
        nonlocal exit_notional, exit_qty_accum, trades_count, op_pnl_points

        if not all([op_open_time, op_direction, open_symbol, pref]):
            return
        if exit_qty_accum <= 0:
            return

        avg_exit = exit_notional / exit_qty_accum
        duration = (close_time - op_open_time).total_seconds()
        pnl_money = (op_pnl_points * point_value) if point_value is not None else None

        ops.append(OperationOut(
            symbol_prefix=pref,
            symbol=open_symbol,
            direction="LONG" if op_direction == "LONG" else "SHORT",
            qty=exit_qty_accum,
            open_time=op_open_time,
            close_time=close_time,
            duration_sec=duration,
            avg_entry_price=avg_entry,
            avg_exit_price=avg_exit,
            pnl_points=op_pnl_points,
            pnl_money=pnl_money,
            trades_count=trades_count,
        ))

        op_open_time = None
        op_direction = None
        open_symbol = None
        pref = None
        exit_notional = 0.0
        exit_qty_accum = 0.0
        trades_count = 0
        op_pnl_points = 0.0

    for o in orders:
        dqty = signed_qty(o.side, float(o.qty))
        trades_count += 1

        if pos_qty == 0:
            pos_qty = dqty
            avg_entry = o.price
            op_open_time = o.timestamp
            op_direction = "LONG" if pos_qty > 0 else "SHORT"
            open_symbol = o.symbol
            pref = symbol_prefix(o.symbol)
            continue

        # aumenta posição (faz preço médio)
        if (pos_qty > 0 and dqty > 0) or (pos_qty < 0 and dqty < 0):
            new_qty = pos_qty + dqty
            avg_entry = (abs(pos_qty) * avg_entry + abs(dqty) * o.price) / abs(new_qty)
            pos_qty = new_qty
            continue

        # fechamento parcial/total
        close_abs = close_qty_portion(pos_qty, dqty)

        if pos_qty > 0:
            realized = close_abs * (o.price - avg_entry)
        else:
            realized = close_abs * (avg_entry - o.price)

        op_pnl_points += realized
        exit_notional += close_abs * o.price
        exit_qty_accum += close_abs

        # parcial
        if abs(dqty) < abs(pos_qty):
            pos_qty = pos_qty + dqty
            continue

        # zerou
        if abs(dqty) == abs(pos_qty):
            pos_qty = 0.0
            finalize_op(close_time=o.timestamp)
            avg_entry = 0.0
            continue

        # inversão (fecha e abre nova)
        remaining = pos_qty + dqty
        pos_qty = 0.0
        finalize_op(close_time=o.timestamp)

        pos_qty = remaining
        avg_entry = o.price
        op_open_time = o.timestamp
        op_direction = "LONG" if pos_qty > 0 else "SHORT"
        open_symbol = o.symbol
        pref = symbol_prefix(o.symbol)

    open_pos: Optional[OpenPositionOut] = None
    if pos_qty != 0 and op_open_time and open_symbol:
        open_pos = OpenPositionOut(
            symbol_prefix=symbol_prefix(open_symbol),
            symbol=open_symbol,
            direction="LONG" if pos_qty > 0 else "SHORT",
            qty=abs(pos_qty),
            avg_entry_price=avg_entry,
            open_time=op_open_time,
        )

    return _BuildResult(operations=ops, open_position=open_pos)


def module0_process(orders: List[OrderIn], pv_map: Dict[str, float]) -> Module0Response:
    if not orders:
        raise HTTPException(status_code=400, detail="Nenhuma ordem válida após filtro (active/status/side/qty/price).")

    orders.sort(key=lambda x: x.timestamp)

    by_prefix: Dict[str, List[OrderIn]] = {}
    clean: List[CleanOrder] = []

    for o in orders:
        pref = symbol_prefix(o.symbol)
        by_prefix.setdefault(pref, []).append(o)
        clean.append(CleanOrder(
            order_id=o.order_id,
            timestamp=o.timestamp,
            symbol=o.symbol,
            symbol_prefix=pref,
            side=o.side,
            qty=o.qty,
            price=o.price,
            status=o.status,
        ))

    ops: List[OperationOut] = []
    open_positions: List[OpenPositionOut] = []

    for pref, olist in by_prefix.items():
        result = build_operations_for_prefix(olist, point_value=pv_map.get(pref))
        ops.extend(result.operations)
        if result.open_position:
            open_positions.append(result.open_position)

    return Module0Response(
        request_id=str(uuid4()),
        meta={
            "orders_used": len(orders),
            "operations_built": len(ops),
            "open_positions": len(open_positions),
            "prefixes": sorted(by_prefix.keys()),
            "side_map": SIDE_MAP,
        },
        orders_clean=clean,
        operations=ops,
        open_positions=open_positions,
    )


# =========================
# Parsing flexível
# =========================

def _extract_orders(payload: Any) -> Tuple[List[ArenaOrderIn], Dict[str, Any], str]:
    """
    Aceita:
      - [ {ordem}, {ordem} ]
      - { "orders": [ ... ], ...extras }
      - { "body": [ ... ], ...extras }   (n8n)
      - [ { "orders": [ ... ] }, ... ]   (novo)
      - [ { "orders": [ ... ] } ]        (novo)
    Retorna (orders, extras, shape)
    """
    extras: Dict[str, Any] = {}

    # { body: [...] }
    if isinstance(payload, dict) and "body" in payload:
        bp = BodyPayload.model_validate(payload)
        extras = {k: v for k, v in payload.items() if k != "body"}
        return bp.body, extras, "BodyPayload"

    # { orders: [...] }
    if isinstance(payload, dict) and "orders" in payload:
        op = OrdersPayload.model_validate(payload)
        extras = {k: v for k, v in payload.items() if k != "orders"}
        return op.orders, extras, "OrdersPayload"

    # [ { orders: [...] }, { orders: [...] } ]  (novo)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "orders" in payload[0]:
        out: List[ArenaOrderIn] = []
        for item in payload:
            op = OrdersPayload.model_validate(item)
            out.extend(op.orders)
        return out, extras, "List[OrdersPayload]"

    # [ {ordem}, {ordem} ]
    if isinstance(payload, list):
        out = [ArenaOrderIn.model_validate(x) for x in payload]
        return out, extras, "List[ArenaOrderIn]"

    raise HTTPException(status_code=400, detail=f"Formato inválido. type={type(payload)}")


# =========================
# FastAPI
# =========================

app = FastAPI(title="Trader Analysis MVP", version="0.2.1")


@app.get("/")
def ping():
    return {"mensagem": "API ativa. Use POST / (ou /analyze) para analisar."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/echo")
async def echo(req: Request):
    try:
        payload = await req.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha lendo JSON: {repr(e)}")
    return {
        "payload_type": str(type(payload)),
        "keys": list(payload.keys()) if isinstance(payload, dict) else None,
        "first_item_keys": list(payload[0].keys()) if isinstance(payload, list) and payload and isinstance(payload[0], dict) else None,
        "payload": payload,
    }


async def _analyze_impl(req: Request):
    # lê json
    try:
        payload = await req.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha lendo JSON: {repr(e)}")

    # extrai
    try:
        arena_orders, extras, shape = _extract_orders(payload)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail={
            "msg": "Erro validando payload",
            "payload_type": str(type(payload)),
            "payload_keys": list(payload.keys()) if isinstance(payload, dict) else None,
            "errors": ve.errors(),
        })

    # converte e conta ignoradas
    internal: List[OrderIn] = []
    ignored = 0
    ignored_reasons = {
        "inactive": 0,
        "not_executed": 0,
        "side_invalid": 0,
        "qty_or_price_invalid": 0,
    }

    for ao in arena_orders:
        if not ao.active:
            ignored += 1
            ignored_reasons["inactive"] += 1
            continue

        if not is_executed(ao):
            ignored += 1
            ignored_reasons["not_executed"] += 1
            continue

        s = str(ao.side).strip().upper()
        if s not in SIDE_MAP:
            ignored += 1
            ignored_reasons["side_invalid"] += 1
            continue

        try:
            qty = float(ao.quantity)
            price = float(ao.price)
        except Exception:
            ignored += 1
            ignored_reasons["qty_or_price_invalid"] += 1
            continue

        if qty <= 0 or price <= 0:
            ignored += 1
            ignored_reasons["qty_or_price_invalid"] += 1
            continue

        oi = arena_to_orderin(ao)
        if oi:
            internal.append(oi)
        else:
            ignored += 1

    # valores de ponto (fixo)
    pv_map = {
        "WIN": 0.2,
        "WDO": 10.0,
        "BIT": 0.2,
    }

    resp = module0_process(internal, pv_map=pv_map)
    resp.meta["orders_ignored"] = ignored
    resp.meta["ignored_reasons"] = ignored_reasons
    resp.meta["payload_shape"] = shape
    resp.meta["extras_keys"] = list(extras.keys())
    return resp


@app.post("/", response_model=Module0Response)
async def analyze_root(req: Request):
    return await _analyze_impl(req)


@app.post("/analyze", response_model=Module0Response)
async def analyze(req: Request):
    return await _analyze_impl(req)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "msg": "Erro interno",
            "path": str(request.url.path),
            "error": repr(exc),
            "request_id": str(uuid4()),
        },
    )
