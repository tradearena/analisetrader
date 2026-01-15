# module0_fastapi.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

Side = Literal["BUY", "SELL"]
OrderStatus = Literal["FILLED"]


# =========================
# Arena input (único formato aceito)
# =========================

class ArenaOrderIn(BaseModel):
    id: str
    code: str
    side: str          # "0" ou "1"
    price: str
    active: bool
    status: str        # "1" = executada
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

    @field_validator("dateTime")
    @classmethod
    def ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


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
# Output
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


SIDE_MAP = {"0": "BUY", "1": "SELL"}
VALID_STATUS = {"1"}


def arena_to_orderin(a: ArenaOrderIn) -> Optional[OrderIn]:
    if not a.active:
        return None
    if a.status not in VALID_STATUS:
        return None

    side = SIDE_MAP.get(a.side)
    if not side:
        return None

    try:
        qty = float(a.quantity)
        price = float(a.price)
    except ValueError:
        return None

    if qty <= 0 or price <= 0:
        return None

    return OrderIn(
        order_id=a.id,
        timestamp=a.dateTime,
        symbol=a.code,
        side=side,
        qty=qty,
        price=price,
        status="FILLED",
    )


def signed_qty(side: Side, qty: float) -> float:
    return qty if side == "BUY" else -qty


def close_qty_portion(pos_qty: float, delta_qty: float) -> float:
    return min(abs(pos_qty), abs(delta_qty))


# =========================
# Core – gera operações
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

        # aumento posição
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

        if abs(dqty) < abs(pos_qty):
            pos_qty = pos_qty + dqty
            continue

        if abs(dqty) == abs(pos_qty):
            pos_qty = 0.0
            finalize_op(close_time=o.timestamp)
            avg_entry = 0.0
            continue

        # inversão
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
        raise HTTPException(status_code=400, detail="Nenhuma ordem válida após filtro.")

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
# FastAPI
# =========================

app = FastAPI(title="Trader Analysis MVP", version="0.1.0")


@app.post("/analyze", response_model=Module0Response)
async def analyze(arena_orders: List[ArenaOrderIn]):
    """
    Aceita APENAS array no formato Arena/Nelogica.
    """
    internal: List[OrderIn] = []
    ignored = 0

    for ao in arena_orders:
        oi = arena_to_orderin(ao)
        if oi:
            internal.append(oi)
        else:
            ignored += 1

    # se quiser, já fixa valores de ponto aqui
    pv_map = {
        # "WIN": 0.2,
        # "WDO": 10.0,
    }

    resp = module0_process(internal, pv_map=pv_map)
    resp.meta["orders_ignored"] = ignored
    return resp
