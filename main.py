from fastapi import FastAPI, Request
import pandas as pd
import traceback
import numpy as np

app = FastAPI()

# Fatores por PREFIXO
multiplicadores = {'WIN': 0.2, 'WDO': 10.0, 'BIT': 0.1}
emolumentos     = {'WIN': 0.25, 'WDO': 1.20, 'BIT': 3.00}

def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def map_side(val):
    if val is None:
        return np.nan
    s = str(val).strip().upper()
    if s in {"0", "BUY", "COMPRA", "B"}:
        return "COMPRA"
    if s in {"1", "SELL", "VENDA", "S"}:
        return "VENDA"
    return np.nan

@app.get("/")
def ping():
    return {"mensagem": "API ativa. Use POST / ou /calcular-resultado"}

@app.post("/")
async def calcular_raiz(request: Request):
    return await calcular_resultado(request)

@app.post("/calcular-resultado")
async def calcular_resultado(request: Request):
    try:
        json_data = await request.json()

        if isinstance(json_data, list):
            orders = json_data
            extra = {}
        elif isinstance(json_data, dict) and "orders" in json_data:
            orders = json_data["orders"]
            extra  = {k: v for k, v in json_data.items() if k != "orders"}
        else:
            return {"erro": "Formato de JSON inválido. Envie um array ou um objeto com 'orders'."}

        if not orders:
            return {"erro": "Lista de ordens vazia."}

    except Exception as e:
        return {"erro": f"Erro ao processar JSON: {str(e)}"}

    try:
        df = pd.DataFrame(orders)

        # Normalizações
        df["dateTime"] = pd.to_datetime(df.get("dateTime"), errors="coerce")
        df["side"]     = df.get("side").apply(map_side)
        df["code"]     = df.get("code").astype(str).str.upper()
        df["token"]    = df.get("token").astype(str)
        df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce")
        df["price"]    = pd.to_numeric(df.get("price"), errors="coerce")
        df["prefix"]   = df["code"].str[:3]

        df = df.dropna(subset=["quantity", "price", "dateTime", "side", "prefix"])

        # ---------- Último preço por PREFIXO ----------
        # 1) Se vier no body (opcional), usa lastPricesPrefix (ex.: {"WDO": 5488, "WIN": 142215})
        last_body = {}
        if isinstance(extra, dict) and "lastPricesPrefix" in extra and isinstance(extra["lastPricesPrefix"], dict):
            last_body = {str(k).upper(): float(v) for k, v in extra["lastPricesPrefix"].items() if v is not None}

        # 2) Último preço do TOKEN "1" por prefixo (no payload)
        df_sorted = df.sort_values("dateTime")
        df_t1 = df_sorted[df_sorted["token"] == "1"].copy()
        ultimo_prefix_token1 = (
            df_t1.groupby("prefix")["price"].last().to_dict()
            if not df_t1.empty else {}
        )

        # 3) Fallback: último preço por prefixo (todos tokens) no payload
        ultimo_prefix_all = df_sorted.groupby("prefix")["price"].last().to_dict()

        def get_last_price_prefix(prefix: str) -> float:
            p = (prefix or "").upper()
            if p in last_body:
                return last_body[p]
            if p in ultimo_prefix_token1:
                return ultimo_prefix_token1[p]
            return ultimo_prefix_all.get(p, 0.0)

        resultados = []

        # ---------- Loop por usuário (token) ----------
        for usuario, g_token in df.groupby("token"):
            lucro_realizado_total = 0.0
            pnl_aberto_total = 0.0
            custo_total = 0.0

            qtde_contratos_total = 0.0
            qtde_compra_total = 0.0
            qtde_venda_total = 0.0
            ordens = len(g_token)

            # ---------- Processa por PREFIXO (mistura maturidades por design) ----------
            for prefix, g_pref in g_token.groupby("prefix"):
                mult = multiplicadores.get(prefix, 0.0)
                taxa_emol = emolumentos.get(prefix, 0.0)

                g_pref = g_pref.sort_values("dateTime")

                # Métricas de volume
                vol_total = g_pref["quantity"].sum()
                qtde_contratos_total += vol_total
                qtde_compra_total += g_pref.loc[g_pref["side"] == "COMPRA", "quantity"].sum()
                qtde_venda_total += g_pref.loc[g_pref["side"] == "VENDA", "quantity"].sum()

                # Emolumentos por volume do prefixo
                custo_total += vol_total * taxa_emol

                # ---- Motor WAC (custo médio móvel) POR PREFIXO ----
                pos = 0.0            # >0 long, <0 short, 0 zerado
                pm = 0.0             # preço médio da posição aberta
                pnl_realizado = 0.0  # acumulado no prefixo

                for _, row in g_pref.iterrows():
                    qty = float(row["quantity"])
                    px = float(row["price"])
                    side = row["side"]

                    if side == "COMPRA":
                        # Fecha short se existir
                        if pos < 0:
                            match_qty = min(qty, -pos)
                            pnl_realizado += (pm - px) * match_qty * mult  # short fecha comprando
                            pos += match_qty
                            qty -= match_qty
                            if pos == 0:
                                pm = 0.0
                        # Abre/aumenta long com eventual sobra
                        if qty > 0:
                            if pos == 0:
                                pm = px
                                pos = qty
                            else:
                                # pos >= 0 aqui
                                pm = (pm * pos + px * qty) / (pos + qty)
                                pos += qty

                    elif side == "VENDA":
                        # Fecha long se existir
                        if pos > 0:
                            match_qty = min(qty, pos)
                            pnl_realizado += (px - pm) * match_qty * mult  # long fecha vendendo
                            pos -= match_qty
                            qty -= match_qty
                            if pos == 0:
                                pm = 0.0
                        # Abre/aumenta short com eventual sobra
                        if qty > 0:
                            if pos == 0:
                                pm = px
                                pos = -qty
                            else:
                                # pos <= 0 aqui
                                abspos = abs(pos)
                                pm = (pm * abspos + px * qty) / (abspos + qty)
                                pos -= qty
                    else:
                        continue

                # ---- PnL aberto (fechamento fantasma) por PREFIXO usando preço do token '1' ----
                ultimo_preco = get_last_price_prefix(prefix)
                pnl_aberto = 0.0
                if pos > 0:
                    pnl_aberto = (ultimo_preco - pm) * pos * mult
                elif pos < 0:
                    pnl_aberto = (pm - ultimo_preco) * (-pos) * mult

                lucro_realizado_total += pnl_realizado
                pnl_aberto_total += pnl_aberto

            lucro_bruto = lucro_realizado_total + pnl_aberto_total
            lucro_liquido = lucro_bruto - custo_total

            resultados.append({
                "token": usuario,
                "lucroBruto": round(lucro_bruto, 2),
                "lucroLiquido": round(lucro_liquido, 2),
                "qtdeOrdens": ordens,
                "qtdeContratos": round(qtde_contratos_total, 2),
                "qtdeCompra": round(qtde_compra_total, 2),
                "qtdeVenda": round(qtde_venda_total, 2)
            })

        # Conversão p/ tipos nativos
        resultados = [{k: convert_numpy(v) for k, v in item.items()} for item in resultados]
        return resultados

    except Exception as e:
        print(traceback.format_exc())
        return {"erro": f"Erro interno durante o cálculo: {str(e)}"}
