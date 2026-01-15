from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
from collections import defaultdict
import math
from typing import Any

app = FastAPI(title="Trader Performance Analyzer")

# =========================
# CONFIG
# =========================

MULTIPLIER = {
    "WIN": 0.2,
    "WDO": 10.0,
    "BIT": 0.2
}

REENTRY_WINDOW = timedelta(minutes=5)
EPS = 1e-9

# =========================
# HELPERS
# =========================

def normalize_asset(code: str):
    code = (code or "").upper()
    if code.startswith("WIN"): return "WIN"
    if code.startswith("WDO"): return "WDO"
    if code.startswith("BIT"): return "BIT"
    return None

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0

def parse_datetime(dt):
    if isinstance(dt, datetime):
        return dt
    s = str(dt)
    # aceita "2026-01-14T15:22:05.000Z"
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    # garante ISO
    s = s.replace(" ", "T", 1)
    return datetime.fromisoformat(s)

def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def extract_orders(payload: Any):
    """
    Aceita:
    1) {"orders":[...]}  (antigo)
    2) [{"orders":[...]}, {"orders":[...]}]  (novo)
    3) {"data":[{"orders":[...]}]} (extra opcional)
    Retorna lista flat de ordens
    """
    # caso 1: dict com orders
    if isinstance(payload, dict):
        if "orders" in payload and isinstance(payload["orders"], list):
            return payload["orders"]

        # opcional: se vier aninhado em "data"
        if "data" in payload and isinstance(payload["data"], list):
            flat = []
            for item in payload["data"]:
                if isinstance(item, dict) and isinstance(item.get("orders"), list):
                    flat.extend(item["orders"])
            return flat

        raise HTTPException(status_code=400, detail="Payload dict inválido: esperado chave 'orders' ou 'data'.")

    # caso 2: lista de blocos com orders
    if isinstance(payload, list):
        flat = []
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("orders"), list):
                flat.extend(item["orders"])
            else:
                raise HTTPException(status_code=400, detail="Payload lista inválido: cada item deve ter {orders:[...]}.")
        return flat

    raise HTTPException(status_code=400, detail="Payload inválido: esperado dict ou list.")

# =========================
# AGREGAÇÃO DE OPERAÇÕES
# =========================

def aggregate_operations(orders):
    grouped = defaultdict(list)

    for o in orders:
        asset = normalize_asset(o.get("code"))
        if not asset:
            continue

        # considera só executadas status=0
        if int(o.get("status", 0)) != 0:
            continue

        grouped[(o.get("tournamentPersonId") or o.get("personId") or "unknown", asset)].append({
            "asset": asset,
            "side": int(o.get("side", 0)),               # 0 buy, 1 sell
            "price": to_float(o.get("price", 0.0)),
            "quantity": to_float(o.get("quantity", 0.0)),
            "dateTime": parse_datetime(o.get("dateTime"))
        })

    operations = []

    for (trader_id, asset), group in grouped.items():
        group.sort(key=lambda x: x["dateTime"])
        position = 0.0
        op = None
        prices, ev, ex, eq, exq = [], 0.0, 0.0, 0.0, 0.0

        for o in group:
            delta = o["quantity"] if o["side"] == 0 else -o["quantity"]
            new_pos = position + delta

            # abre operação quando sai de 0
            if abs(position) < EPS and abs(new_pos) >= EPS:
                op = {
                    "trader_id": trader_id,
                    "asset": asset,
                    "direction": "LONG" if new_pos > 0 else "SHORT",
                    "start_time": o["dateTime"]
                }
                prices, ev, ex, eq, exq = [], 0.0, 0.0, 0.0, 0.0

            if op:
                prices.append(o["price"])
                is_entry = (
                    (op["direction"] == "LONG" and o["side"] == 0) or
                    (op["direction"] == "SHORT" and o["side"] == 1)
                )

                if is_entry:
                    ev += o["price"] * o["quantity"]
                    eq += o["quantity"]
                else:
                    ex += o["price"] * o["quantity"]
                    exq += o["quantity"]

            # inversão: fecha atual e abre outra
            if abs(position) >= EPS and sign(position) != sign(new_pos) and abs(new_pos) >= EPS and op:
                operations.append(finalize_operation(op, prices, ev, ex, eq, exq, o["dateTime"]))
                op = {
                    "trader_id": trader_id,
                    "asset": asset,
                    "direction": "LONG" if new_pos > 0 else "SHORT",
                    "start_time": o["dateTime"]
                }
                prices, ev, ex, eq, exq = [], 0.0, 0.0, 0.0, 0.0

            # fecha quando volta pra 0
            if abs(new_pos) < EPS and op:
                operations.append(finalize_operation(op, prices, ev, ex, eq, exq, o["dateTime"]))
                op = None

            position = new_pos

    mark_reentries(operations)
    return operations

def finalize_operation(op, prices, ev, ex, eq, exq, end_time):
    avg_entry = ev / max(eq, EPS)
    avg_exit = ex / max(exq, EPS)

    if op["direction"] == "LONG":
        pnl_points = (avg_exit - avg_entry) * exq
        diffs = [p - avg_entry for p in prices]
    else:
        pnl_points = (avg_entry - avg_exit) * exq
        diffs = [avg_entry - p for p in prices]

    return {
        "trader_id": op["trader_id"],
        "asset": op["asset"],
        "direction": op["direction"],
        "start_time": op["start_time"].isoformat(),
        "end_time": end_time.isoformat(),
        "duration_sec": (end_time - op["start_time"]).total_seconds(),
        "entry_qty": eq,
        "exit_qty": exq,
        "avg_entry_price": avg_entry,
        "avg_exit_price": avg_exit,
        "pnl_points": pnl_points,
        "pnl": pnl_points * MULTIPLIER[op["asset"]],
        "mfe_points": max(diffs) if diffs else 0.0,
        "mae_points": min(diffs) if diffs else 0.0,
        "reentry_5m": False
    }

def mark_reentries(ops):
    ops.sort(key=lambda x: (x["trader_id"], x["asset"], x["start_time"]))
    last_end = {}

    for o in ops:
        key = (o["trader_id"], o["asset"])
        if key in last_end:
            o["reentry_5m"] = (
                parse_datetime(o["start_time"]) -
                parse_datetime(last_end[key])
            ) <= REENTRY_WINDOW
        last_end[key] = o["end_time"]

# =========================
# MÉTRICAS
# =========================

def compute_summary(ops):
    pnl = [o["pnl"] for o in ops]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]

    equity, acc = [], 0.0
    for p in pnl:
        acc += p
        equity.append(acc)

    max_dd = 0.0
    peak = -1e18
    for v in equity:
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)

    return {
        "total_operations": len(ops),
        "wins": len(wins),
        "losses": len(losses),
        "break_even": len([p for p in pnl if abs(p) < EPS]),
        "net_pnl": sum(pnl),
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses else 0.0,
        "max_drawdown": max_dd
    }

# =========================
# API
# =========================

@app.post("/analyze/trader")
def analyze_trader(payload: Any):
    orders = extract_orders(payload)

    operations = aggregate_operations(orders)
    summary = compute_summary(operations)

    return {
        "executed_orders": len(orders),
        "operations_count": len(operations),
        "summary": summary,
        "operations_sample": sorted(
            operations,
            key=lambda x: abs(x["pnl"]),
            reverse=True
        )[:10]
    }

# Healthcheck simples
@app.get("/health")
def health():
    return {"ok": True}
