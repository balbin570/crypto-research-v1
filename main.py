"""CRYPTO-RESEARCH-V1: read-only research API. No trading endpoints or credentials."""
from datetime import datetime, timezone, timedelta
import math
import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="CRYPTO-RESEARCH-V1", version="0.4.0")
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
    return {"project": "CRYPTO-RESEARCH-V1", "mode": "RESEARCH_PAPER_ONLY", "orders": False, "routes": ["/health", "/mr/signal", "/funding/latest", "/mr/backtest", "/mr/cost-analysis", "/funding/backtest", "/docs"]}

@app.get("/health")
def health():
    return {"status": "OK", "utc": datetime.now(timezone.utc).isoformat(), "paper_only": True, "trading": False, "orders": False, "version": "0.4.0"}

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


# Historical research only: no credentials, balances, or order placement.
# Baseline uses closed bars, next-bar open entry, conservative stop-first handling.

async def historical_candles(symbol: str, interval: str, days: int):
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days + 3)).timestamp() * 1000)
    candles = []
    cursor = start_ms
    async with httpx.AsyncClient(timeout=25) as client:
        while cursor < end_ms:
            try:
                response = await client.get(f"{SPOT}/api/v3/klines", params={
                    "symbol": symbol, "interval": interval, "startTime": cursor,
                    "endTime": end_ms, "limit": 1000})
                response.raise_for_status()
                batch = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise HTTPException(status_code=502, detail=f"Tarihsel veri erişim hatası: {type(exc).__name__}") from exc
            if not isinstance(batch, list) or not batch:
                break
            candles.extend(batch)
            next_cursor = int(batch[-1][0]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 1000:
                break
    dedup = {int(c[0]): c for c in candles if int(c[6]) < end_ms}
    return [dedup[k] for k in sorted(dedup) if k >= int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp()*1000)]


def indicator_adx_atr(highs, lows, closes, i, period=14):
    # Wilder's smoothed TR and directional movement, using data through bar i only.
    if i < period * 2 + 1:
        return None, None
    tr, plus, minus = [], [], []
    for j in range(1, i + 1):
        up = highs[j] - highs[j-1]
        down = lows[j-1] - lows[j]
        plus.append(up if up > down and up > 0 else 0.0)
        minus.append(down if down > up and down > 0 else 0.0)
        tr.append(max(highs[j]-lows[j], abs(highs[j]-closes[j-1]), abs(lows[j]-closes[j-1])))
    atr = sum(tr[:period]) / period
    pdm = sum(plus[:period]) / period
    mdm = sum(minus[:period]) / period
    dx = []
    for k in range(period, len(tr)):
        atr = (atr * (period-1) + tr[k]) / period
        pdm = (pdm * (period-1) + plus[k]) / period
        mdm = (mdm * (period-1) + minus[k]) / period
        pdi = 100*pdm/atr if atr else 0
        mdi = 100*mdm/atr if atr else 0
        dx.append(100*abs(pdi-mdi)/(pdi+mdi) if pdi+mdi else 0)
    if len(dx) < period:
        return None, atr
    adx = sum(dx[:period]) / period
    for v in dx[period:]:
        adx = (adx * (period-1) + v) / period
    return adx, atr


@app.get("/mr/backtest")
async def mr_backtest(
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"] = "BTCUSDT",
    interval: Literal["5m", "15m", "1h"] = "15m",
    days: int = Query(30, ge=7, le=180),
    variant: Literal["A", "B"] = "A",
    fee_bps: float = Query(10.0, ge=0, le=100),
    slippage_bps: float = Query(2.0, ge=0, le=100),
):
    raw = await historical_candles(symbol, interval, days)
    return run_mr_simulation(raw, symbol, interval, days, variant, fee_bps, slippage_bps)


def run_mr_simulation(raw, symbol, interval, days, variant, fee_bps, slippage_bps):
    if len(raw) < 100:
        raise HTTPException(status_code=503, detail="Backtest için yeterli geçmiş mum alınamadı")
    opens = [float(x[1]) for x in raw]
    highs = [float(x[2]) for x in raw]
    lows = [float(x[3]) for x in raw]
    closes = [float(x[4]) for x in raw]
    # Compute ATR and ADX incrementally once; avoids O(N^2) repeated history scans.
    n = len(raw)
    atrs = [None]*n
    adxs = [None]*n
    period = 14
    trs = [0.0]*n; pdms = [0.0]*n; mdms = [0.0]*n
    for j in range(1,n):
        up=highs[j]-highs[j-1]; down=lows[j-1]-lows[j]
        trs[j]=max(highs[j]-lows[j],abs(highs[j]-closes[j-1]),abs(lows[j]-closes[j-1]))
        pdms[j]=up if up>down and up>0 else 0.0
        mdms[j]=down if down>up and down>0 else 0.0
    atr=sum(trs[1:period+1])/period
    pdm=sum(pdms[1:period+1])/period; mdm=sum(mdms[1:period+1])/period
    dx=[]; adx=None
    for j in range(period,n):
        if j>period:
            atr=(atr*(period-1)+trs[j])/period
            pdm=(pdm*(period-1)+pdms[j])/period
            mdm=(mdm*(period-1)+mdms[j])/period
        atrs[j]=atr
        pdi=100*pdm/atr if atr else 0.0
        mdi=100*mdm/atr if atr else 0.0
        d=100*abs(pdi-mdi)/(pdi+mdi) if pdi+mdi else 0.0
        dx.append(d)
        if len(dx)==period:
            adx=sum(dx)/period
        elif len(dx)>period:
            adx=(adx*(period-1)+d)/period
        adxs[j]=adx
    cost=(fee_bps+slippage_bps)/10000
    trades=[]; position=None; equity=1.0; peak=1.0; max_dd=0.0
    for i in range(35,n):
        if position is None and i<n-1:
            ref=closes[i-20:i]
            avg=sum(ref)/20
            sd=math.sqrt(sum((v-avg)**2 for v in ref)/20)
            z=(closes[i]-avg)/sd if sd>0 else 0.0
            if z<=-2 and atrs[i] and (variant=="A" or (adxs[i] is not None and adxs[i]<20)):
                # Signal bar i closed; enter at next bar open (i+1).
                entry_i=i+1
                entry=opens[entry_i]*(1+cost)
                position={"entry_i":entry_i,"entry":entry,"raw_entry":opens[entry_i],
                          "stop":opens[entry_i]-1.5*atrs[i],"target":min(avg,opens[entry_i]+2*atrs[i]),
                          "signal_z":z}
                if position["target"]<=opens[entry_i]:
                    position["target"]=opens[entry_i]+2*atrs[i]
        if position and i>=position["entry_i"]:
            stop=position["stop"]; target=position["target"]
            reason=None; exit_raw=None
            if lows[i]<=stop:
                reason="STOP"; exit_raw=min(opens[i],stop) if i>position["entry_i"] else stop
            elif highs[i]>=target:
                reason="TARGET"; exit_raw=target
            elif i-position["entry_i"]>=12:
                reason="TIME"; exit_raw=closes[i]
            elif i==n-1:
                reason="END_OF_DATA"; exit_raw=closes[i]
            if reason:
                exit_net=exit_raw*(1-cost)
                ret=(exit_net/position["entry"]-1)*100
                equity*=1+ret/100
                peak=max(peak,equity)
                max_dd=max(max_dd,(peak-equity)/peak*100)
                trades.append({"entry_utc":datetime.fromtimestamp(int(raw[position["entry_i"]][0])/1000,timezone.utc).isoformat(),
                               "exit_utc":datetime.fromtimestamp(int(raw[i][6])/1000,timezone.utc).isoformat(),
                               "entry":round(position["entry"],5),"exit":round(exit_net,5),
                               "net_pct":round(ret,5),"exit_reason":reason})
                position=None
    returns=[t["net_pct"] for t in trades]
    wins=[x for x in returns if x>0]; losses=[x for x in returns if x<0]
    pf=sum(wins)/abs(sum(losses)) if losses else (None if not wins else "INF")
    return {"model":"MR_"+variant,"symbol":symbol,"interval":interval,"days":days,
            "paper_only":True,"orders":False,"data_bars":n,
            "assumptions":{"fee_bps_per_side":fee_bps,"slippage_bps_per_side":slippage_bps,
                           "entry":"next bar open after signal","stop_target_same_bar":"stop first",
                           "position_sizing":"one non-overlapping position, compounded full equity; no leverage",
                           "note":"Research simulation; OHLC candles cannot resolve intrabar execution. No live fills."},
            "summary":{"trades":len(trades),"win_rate_pct":round(len(wins)/len(trades)*100,2) if trades else None,
                       "mean_net_pct":round(sum(returns)/len(returns),4) if trades else None,
                       "compounded_return_pct":round((equity-1)*100,4),
                       "profit_factor":round(pf,4) if isinstance(pf,float) else pf,
                       "max_closed_trade_drawdown_pct":round(max_dd,4)},
            "last_10_trades":trades[-10:]}


@app.get("/mr/cost-analysis")
async def mr_cost_analysis(
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"] = "BTCUSDT",
    interval: Literal["5m", "15m", "1h"] = "15m",
    days: int = Query(30, ge=7, le=180),
):
    """Same historical bars for all six scenarios; research only."""
    raw = await historical_candles(symbol, interval, days)
    scenarios = [
        ("zero_cost", 0.0, 0.0),
        ("low_cost", 4.0, 1.0),
        ("baseline_cost", 10.0, 2.0),
    ]
    results = []
    for variant in ("A", "B"):
        for label, fee, slip in scenarios:
            r = run_mr_simulation(raw, symbol, interval, days, variant, fee, slip)
            results.append({
                "model": r["model"], "scenario": label,
                "round_trip_cost_bps": 2 * (fee + slip),
                "trades": r["summary"]["trades"],
                "win_rate_pct": r["summary"]["win_rate_pct"],
                "mean_net_pct": r["summary"]["mean_net_pct"],
                "compounded_return_pct": r["summary"]["compounded_return_pct"],
                "profit_factor": r["summary"]["profit_factor"],
                "max_closed_trade_drawdown_pct": r["summary"]["max_closed_trade_drawdown_pct"],
            })
    return {
        "project": "CRYPTO-RESEARCH-V1", "version": "0.4.0",
        "symbol": symbol, "interval": interval, "days": days,
        "data_bars": len(raw), "paper_only": True, "orders": False,
        "scenarios": results,
        "limitations": [
            "Each cost scenario reruns the same entry/exit logic on the same historical bars.",
            "Execution is OHLC-based and not a tick-level fill simulation.",
            "Stop/target levels are based on raw entry and do not guarantee positive net TARGET trades.",
            "Signals can be generated on a bar where a previous trade exits; only one position at a time.",
            "Max drawdown is computed from closed-trade equity, not intratrade equity.",
            "Research results are not evidence of future profitability."
        ]
    }

# V0.4: historical, read-only cash-and-carry research.
# The strategy uses ONLY funding events already observed before each entry.
async def _funding_history(symbol: str, start_ms: int, end_ms: int):
    out = []
    cursor = start_ms
    async with httpx.AsyncClient(timeout=25) as client:
        while cursor <= end_ms:
            try:
                r = await client.get(f"{FUTURES}/fapi/v1/fundingRate", params={"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000})
                r.raise_for_status()
                batch = r.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise HTTPException(502, f"Funding geçmişi alınamadı: {type(exc).__name__}") from exc
            if not isinstance(batch, list) or not batch:
                break
            out.extend(batch)
            nxt = int(batch[-1]["fundingTime"]) + 1
            if nxt <= cursor:
                break
            cursor = nxt
            if len(batch) < 1000:
                break
    return sorted({int(x["fundingTime"]): (int(x["fundingTime"]), float(x["fundingRate"])) for x in out}.values())


async def _hourly_history(base: str, symbol: str, start_ms: int, end_ms: int, path: str):
    out = {}
    cursor = start_ms
    async with httpx.AsyncClient(timeout=25) as client:
        while cursor < end_ms:
            try:
                r = await client.get(f"{base}{path}", params={"symbol": symbol, "interval": "1h", "startTime": cursor, "endTime": end_ms, "limit": 1000})
                r.raise_for_status()
                batch = r.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise HTTPException(502, f"Saatlik fiyat geçmişi alınamadı: {type(exc).__name__}") from exc
            if not isinstance(batch, list) or not batch:
                break
            for row in batch:
                out[int(row[0])] = float(row[1])  # hourly OPEN, never future close
            nxt = int(batch[-1][0]) + 1
            if nxt <= cursor:
                break
            cursor = nxt
            if len(batch) < 1000:
                break
    return out


@app.get('/funding/backtest')
async def funding_backtest(
    symbol: str = Query('BTCUSDT'),
    days: int = Query(30, ge=7, le=90),
    hold_hours: int = Query(24, ge=8, le=72),
    min_last_funding_pct: float = Query(0.005, ge=0, le=0.2),
    cost_bps_per_leg_side: float = Query(10, ge=0, le=50),
):
    """Simplified cash-and-carry test: spot long + equal-coin perp short.

    One unit of spot entry notional is the PnL denominator. Futures collateral,
    liquidation, borrow, and transfer costs are NOT modeled.
    """
    symbol = symbol.upper()
    check_symbol(symbol)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    hour_ms = 3_600_000
    # Last fully opened hour only; avoid an incomplete interval for end exit.
    end_ms = (now_ms // hour_ms - 1) * hour_ms
    start_ms = end_ms - days * 24 * hour_ms
    warmup_ms = start_ms - 48 * hour_ms
    import asyncio
    spot, perp, events = await asyncio.gather(
        _hourly_history(SPOT, symbol, warmup_ms, end_ms, '/api/v3/klines'),
        _hourly_history(FUTURES, symbol, warmup_ms, end_ms, '/fapi/v1/klines'),
        _funding_history(symbol, warmup_ms, end_ms),
    )
    common = sorted(t for t in spot.keys() & perp.keys() if start_ms <= t <= end_ms)
    if len(common) < 24 * 5 or not events:
        raise HTTPException(503, 'Yeterli eşleştirilmiş spot/futures/funding verisi yok')
    # Strictly previous funding event at the entry timestamp; never use future rates.
    import bisect
    event_times = [t for t, _ in events]
    trades = []
    i = 0
    while i < len(common):
        entry_t = common[i]
        previous_idx = bisect.bisect_left(event_times, entry_t) - 1
        if previous_idx < 0 or events[previous_idx][1] * 100 < min_last_funding_pct:
            i += 1
            continue
        exit_t = entry_t + hold_hours * hour_ms
        if exit_t not in spot or exit_t not in perp or exit_t > end_ms:
            i += 1
            continue
        # Missing hourly data inside holding period invalidates this candidate.
        if any(t not in spot or t not in perp for t in range(entry_t, exit_t + 1, hour_ms)):
            i += 1
            continue
        s0, s1, f0, f1 = spot[entry_t], spot[exit_t], perp[entry_t], perp[exit_t]
        if min(s0, s1, f0, f1) <= 0:
            i += 1
            continue
        # Equal coin quantity: q=1/s0 so spot entry notional is 1.
        spot_pnl_pct = (s1 - s0) / s0 * 100
        perp_pnl_pct = (f0 - f1) / s0 * 100
        # Funding paid/received by short uses funding-time markPrice proxy;
        # official funding event rate is actual, but historical mark is not supplied.
        # Hourly perp OPEN at or immediately after funding is only an approximation.
        lo = bisect.bisect_right(event_times, entry_t)
        hi = bisect.bisect_right(event_times, exit_t)
        funding_pct = 0.0
        funding_events = 0
        valid = True
        for ft, rate in events[lo:hi]:
            proxy_hour = (ft // hour_ms) * hour_ms
            if proxy_hour not in perp:
                valid = False
                break
            funding_pct += rate * perp[proxy_hour] / s0 * 100
            funding_events += 1
        if not valid:
            i += 1
            continue
        # Four fills: spot buy/sell and perp sell/buy; cost on each leg's notional.
        fee_pct = (cost_bps_per_leg_side / 100) * (1 + s1/s0 + f0/s0 + f1/s0)
        net_pct = spot_pnl_pct + perp_pnl_pct + funding_pct - fee_pct
        trades.append({
            'entry_utc': datetime.fromtimestamp(entry_t/1000, timezone.utc).isoformat(),
            'exit_utc': datetime.fromtimestamp(exit_t/1000, timezone.utc).isoformat(),
            'last_known_funding_pct': round(events[previous_idx][1]*100, 6),
            'funding_events': funding_events,
            'spot_pnl_pct': round(spot_pnl_pct, 5),
            'futures_pnl_pct': round(perp_pnl_pct, 5),
            'funding_pnl_pct': round(funding_pct, 5),
            'cost_pct': round(fee_pct, 5),
            'net_pct': round(net_pct, 5),
        })
        i = bisect.bisect_left(common, exit_t) + 1  # no overlap
    vals = [t['net_pct'] for t in trades]
    gross = [t['spot_pnl_pct'] + t['futures_pnl_pct'] + t['funding_pnl_pct'] for t in trades]
    wins = sum(v > 0 for v in vals)
    pos = sum(v for v in vals if v > 0)
    neg = -sum(v for v in vals if v < 0)
    equity, peak, dd = 1.0, 1.0, 0.0
    for v in vals:
        equity *= 1 + v/100
        peak = max(peak, equity)
        dd = max(dd, (peak-equity)/peak*100)
    return {
        'project': 'CRYPTO-RESEARCH-V1', 'version': '0.4.0', 'model': 'FA_A_HISTORICAL',
        'symbol': symbol, 'days': days, 'hold_hours': hold_hours,
        'paper_only': True, 'orders': False,
        'matched_hourly_bars': len(common), 'historical_funding_events': len(events),
        'parameters': {'min_last_funding_pct': min_last_funding_pct,
                       'cost_bps_per_leg_side': cost_bps_per_leg_side},
        'summary': {
            'trades': len(trades), 'win_rate_pct': round(wins/len(vals)*100, 2) if vals else None,
            'mean_gross_pct': round(sum(gross)/len(gross), 5) if gross else None,
            'mean_net_pct': round(sum(vals)/len(vals), 5) if vals else None,
            'compounded_return_pct': round((equity-1)*100, 4) if vals else None,
            'profit_factor': round(pos/neg, 4) if neg else None,
            'max_closed_trade_drawdown_pct': round(dd, 4) if vals else None,
            'total_funding_pct': round(sum(t['funding_pnl_pct'] for t in trades), 5),
            'total_basis_pnl_pct': round(sum(t['spot_pnl_pct'] + t['futures_pnl_pct'] for t in trades), 5),
            'total_cost_pct': round(sum(t['cost_pct'] for t in trades), 5),
        },
        'last_10_trades': trades[-10:],
        'limitations': [
            'Funding rate is historical; payment notional uses an approximate hourly futures price, not historical funding mark.',
            'Entry signal uses only the last realized funding rate, not a known future rate.',
            'Spot and futures hourly open prices are assumed simultaneously executable; spread, depth and fill quality are not verified.',
            'Funding cash flow is excluded at exact entry timestamp; funding after entry through exit is included.',
            'No futures margin, liquidation, borrowing, funding settlement timing or intratrade drawdown model.',
            'Return is normalized to spot notional, not total cash/collateral committed.',
            'Compounded return is a research metric, not a realizable portfolio equity curve.',
            'No live trading, no orders, no evidence of future profitability.',
        ],
    }

# Ensure the old routes remain unchanged. Version is informational only.
app.version = '0.4.0'
