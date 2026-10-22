"""CRYPTO-RESEARCH-V1: read-only research API. No trading endpoints or credentials."""
from datetime import datetime, timezone
import math
import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="CRYPTO-RESEARCH-V1", version="0.1.0")
SPOT = "https://data-api.binance.vision"
FUTURES = "https://fapi.binance.com"
SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
INTERVALS = {"5m", "15m", "1h"}

async def fetch_json(url: str, params: dict):
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Piyasa veri kaynağına erişilemedi: {type(exc).__name__}") from exc

def check_symbol(symbol: str):
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail="Desteklenen semboller: BTCUSDT, ETHUSDT, SOLUSDT")

@app.get("/")
def root():
    return {"project": "CRYPTO-RESEARCH-V1", "mode": "RESEARCH_PAPER_ONLY", "orders": False, "routes": ["/health", "/mr/signal", "/funding/latest", "/docs"]}

@app.get("/health")
def health():
    return {"status": "OK", "utc": datetime.now(timezone.utc).isoformat(), "paper_only": True, "trading": False, "orders": False, "version": "0.1.0"}

@app.get("/mr/signal")
async def mr_signal(symbol: str = Query("BTCUSDT"), interval: Literal["5m", "15m", "1h"] = "15m"):
    symbol = symbol.upper()
    check_symbol(symbol)
    raw = await fetch_json(f"{SPOT}/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": 102})
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    completed = [r for r in raw if int(r[6]) < now_ms]
    closes = [float(r[4]) for r in completed]
    if len(closes) < 40:
        raise HTTPException(status_code=503, detail="Yeterli kapanmış mum yok")
    # Rolling 20-bar z-score. Current close is excluded from reference window.
    reference = closes[-21:-1]
    mean = sum(reference) / len(reference)
    variance = sum((x - mean) ** 2 for x in reference) / len(reference)
    std = math.sqrt(variance)
    z = (closes[-1] - mean) / std if std > 0 else None
    signal = "WATCH_LONG" if z is not None and z <= -2 else "NO_SIGNAL"
    return {"symbol": symbol, "interval": interval, "last_closed_candle_utc": datetime.fromtimestamp(int(completed[-1][6]) / 1000, tz=timezone.utc).isoformat(), "close": closes[-1], "reference_mean_20": round(mean, 8), "z_score": round(z, 4) if z is not None else None, "signal": signal, "note": "Araştırma sinyali; işlem emri değildir. ADX/ATR filtreleri henüz uygulanmadı.", "paper_only": True}

@app.get("/funding/latest")
async def funding_latest(symbol: str = Query("BTCUSDT")):
    symbol = symbol.upper()
    check_symbol(symbol)
    raw = await fetch_json(f"{FUTURES}/fapi/v1/premiumIndex", {"symbol": symbol})
    rate = float(raw["lastFundingRate"])
    return {"symbol": symbol, "last_funding_rate_pct": round(rate * 100, 6), "mark_price": float(raw["markPrice"]), "index_price": float(raw["indexPrice"]), "next_funding_time_utc": datetime.fromtimestamp(int(raw["nextFundingTime"]) / 1000, tz=timezone.utc).isoformat(), "note": "Canlı funding göstergesi; gerçekleşmiş arbitraj kârı değildir. Bölgesel API erişimi değişebilir.", "paper_only": True}
