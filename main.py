"""CRYPTO-RESEARCH-V1: read-only research API. No trading endpoints or credentials."""
from datetime import datetime, timezone, timedelta
import math
import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="CRYPTO-RESEARCH-V1", version="1.5.0")
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
    return {"project": "CRYPTO-RESEARCH-V1", "mode": "RESEARCH_PAPER_ONLY", "orders": False, "routes": ["/health", "/mr/signal", "/funding/latest", "/mr/backtest", "/mr/cost-analysis", "/funding/backtest", "/funding/carry-analysis", "/funding/carry-basis-analysis", "/funding/carry-backtest", "/funding/archive-status", "/funding/reversal-risk", "/docs"]}

@app.get("/health")
def health():
    return {"status": "OK", "utc": datetime.now(timezone.utc).isoformat(), "paper_only": True, "trading": False, "orders": False, "version": "0.9.0"}

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
        "project": "CRYPTO-RESEARCH-V1", "version": "0.9.0",
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
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                source = "SPOT" if base == SPOT else "FUTURES" if base == FUTURES else "UNKNOWN"
                raise HTTPException(502, detail={"error": "Saatlik fiyat geçmişi alınamadı", "source": source, "upstream_http_status": status, "exception": type(exc).__name__, "hint": "403/451: bölgesel erişim; 429: hız sınırı; 400: istek parametresi"}) from exc
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
app.version = '0.7.0'


# V0.5: Funding carry diagnostics, strictly descriptive and research-only.
# All rolling windows are backward-looking, ending at a realized funding event.
@app.get("/funding/carry-analysis")
async def funding_carry_analysis(
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"] = "BTCUSDT",
    days: int = Query(180, ge=30, le=365),
    cost_bps_per_leg_side: float = Query(10.0, ge=0, le=50),
):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 86_400_000
    # 14 days of extra history so the earliest windows can be fully formed.
    events = await _funding_history(symbol, start_ms - 14 * 86_400_000, now_ms)
    events = [(t, r) for t, r in events if t < now_ms]
    in_period = [(t, r) for t, r in events if t >= start_ms]
    if len(in_period) < 15:
        raise HTTPException(503, "Funding analizi için yeterli tarihsel kayıt yok")

    def round_or_none(value, digits=6):
        return round(value, digits) if value is not None else None

    positive = sum(r > 0 for _, r in in_period)
    negative = sum(r < 0 for _, r in in_period)
    zero = len(in_period) - positive - negative
    streaks = []
    streak_sign, streak_len = None, 0
    for _, r in in_period:
        sign = "positive" if r > 0 else "negative" if r < 0 else "zero"
        if sign == streak_sign:
            streak_len += 1
        else:
            if streak_sign is not None:
                streaks.append((streak_sign, streak_len))
            streak_sign, streak_len = sign, 1
    if streak_sign is not None:
        streaks.append((streak_sign, streak_len))

    # Funding rate is a fraction; sum * 100 expresses percentage points of
    # approximate funding return on a constant perp notional. This does NOT
    # include changes in mark notional, basis, fees, or liquidation risk.
    windows = []
    for window_days in (3, 7, 14):
        duration_ms = window_days * 86_400_000
        observations = []
        for end_t, _ in in_period:
            left = end_t - duration_ms
            subset = [(t, r) for t, r in events if left < t <= end_t]
            # Require at least one event per day; do not interpret missing
            # history as zero funding. Intervals vary by instrument/time.
            if len(subset) < window_days:
                continue
            # Reject windows with unexplained gaps longer than 36 hours.
            timestamps = [t for t, _ in subset]
            if any(b - a > 36 * 3_600_000 for a, b in zip(timestamps, timestamps[1:])):
                continue
            total_pct = sum(r for _, r in subset) * 100
            observations.append((end_t, total_pct, len(subset)))
        values = [x[1] for x in observations]
        avg = sum(values) / len(values) if values else None
        latest = observations[-1] if observations else None
        # 4 sides x spot/perp notional assumed approximately equal.
        assumed_round_trip_cost_pct = 4 * cost_bps_per_leg_side / 100
        windows.append({
            "days": window_days,
            "completed_rolling_windows": len(values),
            "mean_cumulative_funding_pct": round_or_none(avg),
            "median_cumulative_funding_pct": round_or_none(sorted(values)[len(values)//2] if values else None),
            "min_cumulative_funding_pct": round_or_none(min(values) if values else None),
            "max_cumulative_funding_pct": round_or_none(max(values) if values else None),
            "positive_window_pct": round_or_none(100 * sum(v > 0 for v in values) / len(values), 2) if values else None,
            "latest_window_end_utc": datetime.fromtimestamp(latest[0]/1000, timezone.utc).isoformat() if latest else None,
            "latest_window_funding_pct": round_or_none(latest[1]) if latest else None,
            "latest_window_funding_events": latest[2] if latest else None,
            "illustrative_four_side_cost_pct": round(assumed_round_trip_cost_pct, 6),
            "latest_funding_minus_cost_pct": round_or_none(latest[1] - assumed_round_trip_cost_pct) if latest else None,
        })
    return {
        "project": "CRYPTO-RESEARCH-V1", "version": "0.9.0",
        "model": "FA_B_FUNDING_PERSISTENCE_DIAGNOSTIC",
        "symbol": symbol, "days": days, "paper_only": True, "orders": False,
        "period_funding_events": len(in_period),
        "funding_signs": {"positive": positive, "negative": negative, "zero": zero,
                          "positive_pct": round(100 * positive / len(in_period), 2)},
        "streaks_in_events": {
            "longest_positive": max((n for sign, n in streaks if sign == "positive"), default=0),
            "longest_negative": max((n for sign, n in streaks if sign == "negative"), default=0),
        },
        "rolling_windows": windows,
        "parameters": {"cost_bps_per_leg_side": cost_bps_per_leg_side},
        "limitations": [
            "Descriptive retrospective funding diagnostics, NOT a tradable backtest or a forward signal.",
            "Rolling windows overlap heavily; their observations are not independent.",
            "Cumulative funding assumes constant perpetual notional and historical realized rates.",
            "Cost is illustrative: four fills at the same per-leg per-side bps, without dynamic notional.",
            "No spot/perp basis PnL, spreads, slippage, borrow, margin, liquidation or collateral modeling.",
            "Funding settlement intervals can vary; data completeness is approximated by event count and gaps.",
            "A positive funding-minus-cost figure does not establish arbitrage profitability.",
        ],
    }


# Daily price history minimizes Binance requests for long-window research.
async def _daily_history(base: str, symbol: str, start_ms: int, end_ms: int, path: str):
    source = "SPOT" if base == SPOT else "FUTURES"
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.get(f"{base}{path}", params={
                "symbol": symbol, "interval": "1d", "startTime": start_ms,
                "endTime": end_ms, "limit": 1000,
            })
            response.raise_for_status()
            batch = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        raise HTTPException(503, detail={
            "error": "Günlük fiyat verisi alınamadı", "source": source,
            "upstream_http_status": status,
            "note": "HTTP 418: Binance geçici IP engeli; yeniden denemeleri durdurup engelin kalkmasını bekleyin."
                    if status == 418 else "Veri kaynağı isteği başarısız oldu."
        }) from exc
    if not isinstance(batch, list):
        raise HTTPException(502, detail=f"{source} beklenmeyen veri biçimi")
    return {int(row[0]): float(row[1]) for row in batch if int(row[0]) <= end_ms}


# V0.6: Historical spot/perpetual carry with realized funding and basis PnL.
# This is a retrospective rolling-window diagnostic, NOT a forward-entry backtest.
@app.get("/funding/carry-basis-analysis")
async def funding_carry_basis_analysis(
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"] = "BTCUSDT",
    days: int = Query(180, ge=60, le=180),
    sample_every_hours: int = Query(24, ge=24, le=168),
):
    import asyncio
    import bisect

    hour_ms = 3_600_000
    day_ms = 24 * hour_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Use UTC midnight OPEN for both legs; last completed UTC date only.
    end_ms = (now_ms // day_ms - 1) * day_ms
    start_ms = end_ms - days * day_ms
    spot, perp, events = await asyncio.gather(
        _daily_history(SPOT, symbol, start_ms, end_ms, "/api/v3/klines"),
        _daily_history(FUTURES, symbol, start_ms, end_ms, "/fapi/v1/klines"),
        _funding_history(symbol, start_ms, end_ms),
    )
    common = sorted(set(spot) & set(perp))
    if len(common) < days * 0.9 or len(events) < days:
        raise HTTPException(503, "Yeterli eşleşen günlük spot/perp veya funding geçmişi yok")
    event_times = [t for t, _ in events]
    durations = (7, 14, 30, 45)
    cost_scenarios = [
        ("low_cost", 2.0),       # 4 sides x 2 bps ~ 0.08%
        ("medium_cost", 4.0),    # 4 sides x 4 bps ~ 0.16%
        ("baseline_cost", 10.0), # 4 sides x 10 bps ~ 0.40%
    ]
    analyses = []
    for duration_days in durations:
        duration_ms = duration_days * day_ms
        records = []
        # Each entry is a sampled historical start, not a trading signal.
        for entry_t in range(start_ms, end_ms - duration_ms + 1, max(day_ms, sample_every_hours * hour_ms)):
            exit_t = entry_t + duration_ms
            if entry_t not in spot or entry_t not in perp or exit_t not in spot or exit_t not in perp:
                continue
            # Reject missing interior daily bars, including gaps in either market.
            if any(t not in spot or t not in perp for t in range(entry_t, exit_t + 1, day_ms)):
                continue
            s0, s1, f0, f1 = spot[entry_t], spot[exit_t], perp[entry_t], perp[exit_t]
            if min(s0, s1, f0, f1) <= 0:
                continue
            lo = bisect.bisect_right(event_times, entry_t)
            hi = bisect.bisect_right(event_times, exit_t)
            funding_events = events[lo:hi]
            if len(funding_events) < duration_days:
                continue
            # Do not treat missing funding data as zero payments.
            times = [entry_t] + [t for t, _ in funding_events] + [exit_t]
            if any(b - a > 36 * hour_ms for a, b in zip(times, times[1:])):
                continue
            funding_pct = 0.0
            for t, rate in funding_events:
                proxy_hour = (t // day_ms) * day_ms
                if proxy_hour not in perp:
                    funding_pct = None
                    break
                funding_pct += 100 * rate * perp[proxy_hour] / s0
            if funding_pct is None:
                continue
            # Equal coin quantity; returns normalized to entry spot notional.
            basis_pct = 100 * ((s1 - s0) + (f0 - f1)) / s0
            gross_pct = funding_pct + basis_pct
            # Fee model accounts for changing notional on all four fills.
            four_sides_notional = 1 + s1/s0 + f0/s0 + f1/s0
            records.append({
                "entry_utc": datetime.fromtimestamp(entry_t/1000, timezone.utc).isoformat(),
                "exit_utc": datetime.fromtimestamp(exit_t/1000, timezone.utc).isoformat(),
                "funding_events": len(funding_events),
                "funding_pct": funding_pct,
                "basis_pnl_pct": basis_pct,
                "gross_pct": gross_pct,
                "cost_multiplier": four_sides_notional,
            })

        scenarios = []
        for label, bps in cost_scenarios:
            values = [r["gross_pct"] - (bps / 100) * r["cost_multiplier"] for r in records]
            scenarios.append({
                "scenario": label,
                "cost_bps_per_leg_side": bps,
                "illustrative_four_side_cost_pct": round(4 * bps / 100, 4),
                "mean_net_pct": round(sum(values)/len(values), 6) if values else None,
                "median_net_pct": round(sorted(values)[len(values)//2], 6) if values else None,
                "positive_net_window_pct": round(100 * sum(v > 0 for v in values)/len(values), 2) if values else None,
                "min_net_pct": round(min(values), 6) if values else None,
                "max_net_pct": round(max(values), 6) if values else None,
            })
        analyses.append({
            "hold_days": duration_days,
            "sampled_windows": len(records),
            "mean_funding_pct": round(sum(r["funding_pct"] for r in records)/len(records), 6) if records else None,
            "mean_basis_pnl_pct": round(sum(r["basis_pnl_pct"] for r in records)/len(records), 6) if records else None,
            "mean_gross_pct": round(sum(r["gross_pct"] for r in records)/len(records), 6) if records else None,
            "cost_scenarios": scenarios,
            "latest_window": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in records[-1].items() if k != "cost_multiplier"
            } if records else None,
        })
    return {
        "project": "CRYPTO-RESEARCH-V1",
        "version": "0.9.0",
        "model": "FA_B_RETROSPECTIVE_CARRY_BASIS",
        "symbol": symbol,
        "days": days,
        "sample_every_hours": sample_every_hours,
        "paper_only": True,
        "orders": False,
        "matched_daily_bars": len(common),
        "historical_funding_events": len(events),
        "analyses": analyses,
        "limitations": [
            "Retrospective sampled windows; no forward signal or tradable entry rule.",
            "Windows overlap and are not independent; do not compound their returns.",
            "Funding uses historical realized rates and DAILY perp-open notional proxy, not exact funding mark.",
            "Spot and perpetual hourly opens are not guaranteed simultaneous executable fills.",
            "Cost scenarios include illustrative fees only; extra spread, slippage and market impact are excluded.",
            "No margin, liquidation, financing, borrow, collateral, transfer or intratrade drawdown model.",
            "Results are normalized to entry spot notional, not total committed capital.",
            "Missing daily data and large funding-event gaps cause windows to be excluded.",
            "No orders, no live trading, no proof of future profitability.",
        ],
    }


# V0.7: chronological, non-overlapping carry backtest.
# The entry decision uses ONLY the previous seven days of realized funding.

# V0.7.1: Binance Vision public monthly archives; no Binance Futures REST calls.
import asyncio as _archive_asyncio
import csv as _archive_csv
import io as _archive_io
import zipfile as _archive_zipfile
from collections import OrderedDict as _ArchiveOrderedDict

_ARCHIVE_ROOT = "https://data.binance.vision/data"
_ARCHIVE_CACHE = _ArchiveOrderedDict()
_ARCHIVE_LOCK = _archive_asyncio.Lock()
_ARCHIVE_MAX_CACHE = 96

def _archive_months(start_ms, end_ms):
    start = datetime.fromtimestamp(start_ms / 1000, timezone.utc)
    end = datetime.fromtimestamp(end_ms / 1000, timezone.utc)
    y, m = start.year, start.month
    result = []
    while (y, m) <= (end.year, end.month):
        result.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return result

def _archive_url(symbol, month, kind):
    if kind == "spot":
        return f"{_ARCHIVE_ROOT}/spot/monthly/klines/{symbol}/1d/{symbol}-1d-{month}.zip"
    if kind == "perp":
        return f"{_ARCHIVE_ROOT}/futures/um/monthly/klines/{symbol}/1d/{symbol}-1d-{month}.zip"
    return f"{_ARCHIVE_ROOT}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"

def _archive_timestamp(value):
    x = int(float(value))
    if x >= 10**14:  # Binance spot klines after 2025 use microseconds
        x //= 1000
    return x

def _archive_parse(content, kind):
    with _archive_zipfile.ZipFile(_archive_io.BytesIO(content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX')]
        if len(names) != 1:
            raise ValueError("Arşivde tek CSV bulunamadı")
        with z.open(names[0]) as f:
            reader = _archive_csv.reader(_archive_io.TextIOWrapper(f, encoding='utf-8-sig'))
            header = next(reader, None)
            if header is None:
                raise ValueError("Boş CSV")
            # Klines may have header or no header.
            if kind != 'funding':
                rows = []
                if header and header[0].strip().isdigit():
                    rows.append(header)
                rows.extend(reader)
                return {_archive_timestamp(row[0]): float(row[1]) for row in rows if len(row) > 1}
            # Funding archive uses calc_time, funding_interval_hours, last_funding_rate.
            normalized = [c.strip().lower().replace(' ', '_') for c in header]
            if 'calc_time' in normalized or 'fundingtime' in normalized:
                t_idx = normalized.index('calc_time') if 'calc_time' in normalized else normalized.index('fundingtime')
                rate_idx = next((normalized.index(k) for k in ('last_funding_rate', 'fundingrate', 'funding_rate') if k in normalized), None)
                if rate_idx is None:
                    raise ValueError('Funding rate sütunu bulunamadı')
                rows = reader
            else:
                # Never guess a funding CSV column mapping.
                raise ValueError(f'Bilinmeyen funding CSV başlıkları: {header[:5]}')
            return {_archive_timestamp(row[t_idx]): float(row[rate_idx]) for row in rows if len(row) > max(t_idx, rate_idx)}

async def _archive_fetch(client, symbol, month, kind):
    key = (symbol, month, kind)
    async with _ARCHIVE_LOCK:
        if key in _ARCHIVE_CACHE:
            _ARCHIVE_CACHE.move_to_end(key)
            return _ARCHIVE_CACHE[key]
    url = _archive_url(symbol, month, kind)
    try:
        response = await client.get(url)
        response.raise_for_status()
        parsed = _archive_parse(response.content, kind)
    except (httpx.HTTPError, ValueError, _archive_zipfile.BadZipFile) as exc:
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        raise HTTPException(503, detail={
            'error': 'Binance arşiv dosyası alınamadı veya okunamadı',
            'dataset': kind, 'month': month, 'http_status': status,
            'exception': type(exc).__name__, 'note': str(exc)[:180] if status is None else 'Arşiv erişimi başarısız'
        }) from exc
    async with _ARCHIVE_LOCK:
        _ARCHIVE_CACHE[key] = parsed
        _ARCHIVE_CACHE.move_to_end(key)
        while len(_ARCHIVE_CACHE) > _ARCHIVE_MAX_CACHE:
            _ARCHIVE_CACHE.popitem(last=False)
    return parsed

async def _archive_bundle(symbol, start_ms, end_ms):
    months = _archive_months(start_ms, end_ms)
    # A small concurrency limit avoids bursts and keeps Render memory bounded.
    async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client:
        spot, perp, funding = {}, {}, {}
        for month in months:
            for kind, target in [('spot', spot), ('perp', perp), ('funding', funding)]:
                part = await _archive_fetch(client, symbol, month, kind)
                target.update(part)
    spot = {t: p for t, p in spot.items() if start_ms <= t <= end_ms}
    perp = {t: p for t, p in perp.items() if start_ms <= t <= end_ms}
    events = sorted((t, r) for t, r in funding.items() if start_ms <= t <= end_ms)
    return spot, perp, events

@app.get('/funding/archive-status')
async def funding_archive_status(symbol: Literal['BTCUSDT', 'ETHUSDT', 'SOLUSDT'] = 'BTCUSDT'):
    now = datetime.now(timezone.utc)
    y, m = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
    month = f'{y:04d}-{m:02d}'
    async with httpx.AsyncClient(timeout=35, follow_redirects=True) as client:
        data = {}
        for kind in ('spot', 'perp', 'funding'):
            part = await _archive_fetch(client, symbol, month, kind)
            data[kind] = {'rows': len(part), 'min_timestamp_ms': min(part) if part else None, 'max_timestamp_ms': max(part) if part else None}
    return {'project': 'CRYPTO-RESEARCH-V1', 'version': '0.9.0', 'source': 'Binance Vision', 'month': month, 'symbol': symbol, 'datasets': data, 'paper_only': True, 'orders': False}

@app.get("/funding/carry-backtest")
async def funding_carry_backtest(
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"] = "BTCUSDT",
    days: int = Query(180, ge=120, le=180),
    lookback_days: int = Query(7, ge=3, le=14),
    min_past_funding_pct: float = Query(0.05, ge=0, le=1),
):
    import asyncio
    import bisect
    DAY = 86_400_000
    HOUR = 3_600_000
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Monthly funding archives are only used through the last completed month.
    today = datetime.now(timezone.utc)
    end = int(datetime(today.year, today.month, 1, tzinfo=timezone.utc).timestamp() * 1000) - DAY
    start = end - days * DAY
    warmup = start - (lookback_days + 2) * DAY
    spot, perp, events = await _archive_bundle(symbol, warmup, end)
    times = [t for t, _ in events]
    if len(spot.keys() & perp.keys()) < days * 0.9 or len(events) < days:
        raise HTTPException(503, "Yeterli tarihsel veri bulunamadı")
    split = start + (days * 2 // 3) * DAY
    cost_scenarios = [("low", 2.0), ("medium", 4.0), ("baseline", 10.0)]
    results = []
    for hold_days in (30, 45):
        trades = []
        entry = start
        while entry + hold_days * DAY <= end:
            exit_t = entry + hold_days * DAY
            if any(t not in spot or t not in perp for t in range(entry, exit_t + DAY, DAY)):
                entry += DAY
                continue
            s0, s1, f0, f1 = spot[entry], spot[exit_t], perp[entry], perp[exit_t]
            if min(s0, s1, f0, f1) <= 0:
                entry += DAY
                continue
            # Known-at-entry trailing realized funding, strictly before entry.
            prior = events[bisect.bisect_left(times, entry - lookback_days * DAY):
                           bisect.bisect_left(times, entry)]
            prior_times = [entry - lookback_days * DAY] + [t for t, _ in prior] + [entry]
            if len(prior) < lookback_days or any(
                b - a > 36 * HOUR for a, b in zip(prior_times, prior_times[1:])
            ):
                entry += DAY
                continue
            trailing_pct = sum(rate * 100 for _, rate in prior)
            if trailing_pct < min_past_funding_pct:
                entry += DAY
                continue
            # Split classification uses entry timestamp, not exit timestamp.
            # Trades crossing split are omitted to keep train/test separated.
            if entry < split < exit_t:
                entry = split
                continue
            actual = events[bisect.bisect_right(times, entry):
                            bisect.bisect_right(times, exit_t)]
            gaps = [entry] + [t for t, _ in actual] + [exit_t]
            if len(actual) < hold_days or any(
                b - a > 36 * HOUR for a, b in zip(gaps, gaps[1:])
            ):
                entry += DAY
                continue
            funding = 0.0
            valid = True
            for t, rate in actual:
                daily_t = (t // DAY) * DAY
                if daily_t not in perp:
                    valid = False
                    break
                funding += 100 * rate * perp[daily_t] / s0
            if not valid:
                entry += DAY
                continue
            basis = 100 * ((s1 - s0) + (f0 - f1)) / s0
            four_fill_notional = 1 + s1/s0 + f0/s0 + f1/s0
            trades.append({
                "entry_utc": datetime.fromtimestamp(entry/1000, timezone.utc).isoformat(),
                "exit_utc": datetime.fromtimestamp(exit_t/1000, timezone.utc).isoformat(),
                "partition": "development" if exit_t <= split else "validation",
                "past_funding_pct_at_entry": round(trailing_pct, 6),
                "realized_funding_pct": round(funding, 6),
                "basis_pnl_pct": round(basis, 6),
                "gross_pct": round(funding+basis, 6),
                "_gross": funding+basis,
                "_notional": four_fill_notional,
            })
            entry = exit_t + DAY  # no overlap, one calendar-day cooldown

        summaries = []
        for partition in ("development", "validation"):
            subset = [t for t in trades if t["partition"] == partition]
            for label, bps in cost_scenarios:
                returns = [t["_gross"] - bps / 100 * t["_notional"] for t in subset]
                equity, peak, max_dd = 1.0, 1.0, 0.0
                for r in returns:
                    equity *= (1 + r / 100)
                    peak = max(peak, equity)
                    max_dd = max(max_dd, (peak - equity) / peak * 100)
                summaries.append({
                    "partition": partition, "cost_scenario": label,
                    "cost_bps_per_leg_side": bps,
                    "trades": len(returns),
                    "mean_net_pct": round(sum(returns)/len(returns), 6) if returns else None,
                    "positive_trade_pct": round(100*sum(r>0 for r in returns)/len(returns), 2) if returns else None,
                    "illustrative_compounded_pct": round(100*(equity-1), 6) if returns else None,
                    "max_closed_trade_drawdown_pct": round(max_dd, 6) if returns else None,
                })
        results.append({
            "hold_days": hold_days,
            "total_nonoverlapping_trades": len(trades),
            "summaries": summaries,
            "trades": [{k: v for k, v in t.items() if not k.startswith("_")} for t in trades],
        })
    return {
        "project": "CRYPTO-RESEARCH-V1", "version": "0.9.0",
        "model": "FA_B_CHRONOLOGICAL_PAPER_BACKTEST",
        "symbol": symbol, "days": days,
        "lookback_days": lookback_days,
        "min_past_funding_pct": min_past_funding_pct,
        "development_validation_split_utc": datetime.fromtimestamp(split/1000, timezone.utc).isoformat(),
        "paper_only": True, "orders": False, "data_source": "Binance Vision monthly ZIP archives", "archive_end_utc": datetime.fromtimestamp(end/1000, timezone.utc).isoformat(), "results": results,
        "limitations": [
            "Entry rule is fixed, uses only realized historical funding known before entry.",
            "Chronological split does not guarantee independent market regimes; very few trades expected.",
            "Parameters and durations were inspired by prior analysis of overlapping windows: validation is NOT fully untouched.",
            "Daily open prices are approximate simultaneous fills; funding notional uses daily perp-open proxy.",
            "Cost model covers illustrative four-fill fees, not extra slippage/spread/market impact.",
            "No intratrade basis drawdown, margin, liquidation, financing, collateral, borrowing or transfer model.",
            "Returns normalized to spot entry notional, not total capital; compounding is illustrative.",
            "Funding is realized after entry; not known in advance. No live orders or profitability guarantee.",
            "Only completed monthly archives are used, so results end at the prior month.",
        ],
    }


# V0.8: diagnostic only. No rule optimization, new trades, or live orders.
@app.get("/funding/reversal-risk")
async def funding_reversal_risk(
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"] = "BTCUSDT",
    days: int = Query(180, ge=120, le=180),
    lookback_days: int = Query(7, ge=3, le=14),
    min_past_funding_pct: float = Query(0.05, ge=0, le=1),
):
    from bisect import bisect_left, bisect_right
    DAY, HOUR = 86_400_000, 3_600_000
    today = datetime.now(timezone.utc)
    end = int(datetime(today.year, today.month, 1, tzinfo=timezone.utc).timestamp()*1000) - DAY
    start = end - days*DAY
    warmup = start - (lookback_days+2)*DAY
    spot, perp, events = await _archive_bundle(symbol, warmup, end)
    times = [t for t, _ in events]
    if len(events) < days or len(spot.keys() & perp.keys()) < days*.9:
        raise HTTPException(503, "Yeterli arşiv verisi bulunamadı")
    base = await funding_carry_backtest(symbol=symbol, days=days,
                                        lookback_days=lookback_days,
                                        min_past_funding_pct=min_past_funding_pct)
    diagnostics = []
    for group in base["results"]:
        for trade in group["trades"]:
            entry = int(datetime.fromisoformat(trade["entry_utc"]).timestamp()*1000)
            exit_t = int(datetime.fromisoformat(trade["exit_utc"]).timestamp()*1000)
            s0, f0 = spot[entry], perp[entry]
            inside = events[bisect_right(times, entry):bisect_right(times, exit_t)]
            daily_path = []
            running_funding = 0.0
            index = 0
            negative_events = sum(rate < 0 for _, rate in inside)
            negative_days = 0
            first_negative_day = None
            worst_daily_funding_pct = None
            for day in range(entry+DAY, exit_t+DAY, DAY):
                day_funding = 0.0
                while index < len(inside) and inside[index][0] <= day:
                    t, rate = inside[index]
                    daily_open = (t//DAY)*DAY
                    if daily_open in perp:
                        amount = 100*rate*perp[daily_open]/s0
                        running_funding += amount
                        day_funding += amount
                    index += 1
                if day not in spot or day not in perp:
                    continue
                if day_funding < 0:
                    negative_days += 1
                    if first_negative_day is None:
                        first_negative_day = (day-entry)//DAY
                worst_daily_funding_pct = (day_funding if worst_daily_funding_pct is None
                                           else min(worst_daily_funding_pct, day_funding))
                basis = 100*((spot[day]-s0)+(f0-perp[day]))/s0
                gross = running_funding + basis
                daily_path.append((day, gross))
            if not daily_path:
                continue
            peak = 0.0
            max_peak_to_trough = 0.0
            for _, gross in daily_path:
                peak = max(peak, gross)
                max_peak_to_trough = max(max_peak_to_trough, peak-gross)
            min_gross = min(v for _, v in daily_path)
            diagnostics.append({
                "hold_days": group["hold_days"],
                "entry_utc": trade["entry_utc"],
                "partition": trade["partition"],
                "negative_funding_events": negative_events,
                "negative_funding_days": negative_days,
                "first_negative_funding_day": first_negative_day,
                "worst_daily_funding_pct": round(worst_daily_funding_pct, 6) if worst_daily_funding_pct is not None else None,
                "min_daily_mark_gross_pct": round(min_gross, 6),
                "max_daily_peak_to_trough_pct_points": round(max_peak_to_trough, 6),
                "final_gross_pct": trade["gross_pct"],
            })
    groups = []
    for hold in (30,45):
        subset = [r for r in diagnostics if r["hold_days"] == hold]
        groups.append({
            "hold_days": hold, "trades": len(subset),
            "trades_with_negative_funding_event": sum(r["negative_funding_events"]>0 for r in subset),
            "trades_with_negative_funding_day": sum(r["negative_funding_days"]>0 for r in subset),
            "mean_negative_funding_days": round(sum(r["negative_funding_days"] for r in subset)/len(subset),3) if subset else None,
            "worst_observed_daily_mark_gross_pct": min((r["min_daily_mark_gross_pct"] for r in subset), default=None),
            "largest_daily_peak_to_trough_pct_points": max((r["max_daily_peak_to_trough_pct_points"] for r in subset), default=None),
        })
    return {
        "project": "CRYPTO-RESEARCH-V1", "version": "0.9.0",
        "model": "FA_B_FUNDING_REVERSAL_RISK_DIAGNOSTIC",
        "paper_only": True, "orders": False, "symbol": symbol, "days": days,
        "source": "Binance Vision completed monthly archives",
        "groups": groups, "trade_diagnostics": diagnostics,
        "limitations": [
            "Uses V0.7.1 chronological entry rules; no new entry/exit strategy is tested.",
            "Negative funding day is defined by net daily funding sum, not every individual event.",
            "Daily marks are synthetic spot-long/perp-short PnL, not executable liquidation values.",
            "Daily prices miss intraday stress; funding uses daily perp-open notional approximation.",
            "No margin, liquidation, collateral, financing, spreads or slippage risk model.",
            "Tiny sample; descriptive diagnostics only, no reliable forecast or proven edge.",
        ],
    }


# V0.9 - Predeclared exit rules; no lookahead and no real orders.
@app.get("/funding/exit-comparison")
async def funding_exit_comparison(
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"] = "BTCUSDT",
    days: int = Query(180, ge=120, le=180),
    lookback_days: int = Query(7, ge=3, le=14),
    min_past_funding_pct: float = Query(0.05, ge=0, le=1),
):
    import bisect
    DAY, HOUR = 86400000, 3600000
    now = datetime.now(timezone.utc)
    end = int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp()*1000) - DAY
    start = end - days*DAY
    split = start + (days*2//3)*DAY
    spot, perp, events = await _archive_bundle(symbol, start-(lookback_days+3)*DAY, end)
    times = [t for t, _ in events]
    if len(events)<days or len(spot.keys() & perp.keys())<days*.9:
        raise HTTPException(503, "Yeterli arşiv verisi bulunamadı")
    output=[]
    rules = ("fixed", "last_3d_negative", "last_7d_negative", "three_negative_days")
    for hold in (30,45):
        for rule in rules:
            trades=[]
            entry=start
            while entry+hold*DAY <= end:
                if entry not in spot or entry not in perp:
                    entry+=DAY; continue
                prior=events[bisect.bisect_left(times,entry-lookback_days*DAY):
                             bisect.bisect_left(times,entry)]
                prior_gaps=[entry-lookback_days*DAY]+[t for t,_ in prior]+[entry]
                if len(prior)<lookback_days or any(b-a>36*HOUR for a,b in zip(prior_gaps,prior_gaps[1:])):
                    entry+=DAY; continue
                trailing=100*sum(r for _,r in prior)
                if trailing<min_past_funding_pct:
                    entry+=DAY; continue
                if entry<split<entry+hold*DAY:
                    entry=split; continue
                s0,f0=spot[entry],perp[entry]
                if min(s0,f0)<=0:
                    entry+=DAY; continue
                exit_t=None
                reason="fixed_hold"
                streak=0
                for d in range(1,hold+1):
                    day=entry+d*DAY
                    if day not in spot or day not in perp:
                        break
                    # Funding events strictly earlier than today's daily open are known.
                    today_events=events[bisect.bisect_left(times,day-DAY):
                                        bisect.bisect_left(times,day)]
                    if len(today_events)<1:
                        break
                    today_sum=sum(rate for _,rate in today_events)
                    streak=streak+1 if today_sum<0 else 0
                    if rule=="last_3d_negative" and d>=3:
                        recent=events[bisect.bisect_left(times,day-3*DAY):
                                      bisect.bisect_left(times,day)]
                        trigger=sum(rate for _,rate in recent)<0
                    elif rule=="last_7d_negative" and d>=7:
                        recent=events[bisect.bisect_left(times,day-7*DAY):
                                      bisect.bisect_left(times,day)]
                        trigger=sum(rate for _,rate in recent)<0
                    elif rule=="three_negative_days":
                        trigger=streak>=3
                    else:
                        trigger=False
                    if trigger:
                        exit_t=day
                        reason=rule
                        break
                    if d==hold:
                        exit_t=day
                if exit_t is None:
                    entry+=DAY; continue
                actual=events[bisect.bisect_right(times,entry):
                              bisect.bisect_left(times,exit_t)]
                gaps=[entry]+[t for t,_ in actual]+[exit_t]
                if len(actual)<(exit_t-entry)//DAY or any(b-a>36*HOUR for a,b in zip(gaps,gaps[1:])):
                    entry+=DAY; continue
                if any((t//DAY)*DAY not in perp for t,_ in actual):
                    entry+=DAY; continue
                funding=sum(100*r*perp[(t//DAY)*DAY]/s0 for t,r in actual)
                s1,f1=spot[exit_t],perp[exit_t]
                basis=100*((s1-s0)+(f0-f1))/s0
                gross=funding+basis
                four=1+s1/s0+f0/s0+f1/s0
                trades.append({
                    "entry_utc":datetime.fromtimestamp(entry/1000,timezone.utc).isoformat(),
                    "exit_utc":datetime.fromtimestamp(exit_t/1000,timezone.utc).isoformat(),
                    "partition":"development" if exit_t<=split else "validation",
                    "days_held":(exit_t-entry)//DAY,
                    "exit_reason":reason,
                    "gross_pct":round(gross,6),
                    "net_low_pct":round(gross-0.02*four,6),
                    "net_medium_pct":round(gross-0.04*four,6),
                    "net_baseline_pct":round(gross-0.1*four,6),
                })
                entry=exit_t+DAY
            summary=[]
            for partition in ("development","validation"):
                subset=[t for t in trades if t["partition"]==partition]
                row={"partition":partition,"trades":len(subset),
                     "early_exits":sum(t["exit_reason"]!="fixed_hold" for t in subset)}
                for cost in ("low","medium","baseline"):
                    vals=[t[f"net_{cost}_pct"] for t in subset]
                    row[f"mean_net_{cost}_pct"]=round(sum(vals)/len(vals),6) if vals else None
                    row[f"positive_{cost}_pct"]=round(100*sum(v>0 for v in vals)/len(vals),2) if vals else None
                    eq=1.0
                    for v in vals: eq*=1+v/100
                    row[f"illustrative_compounded_{cost}_pct"]=round(100*(eq-1),6) if vals else None
                summary.append(row)
            output.append({"hold_days":hold,"exit_rule":rule,
                           "total_trades":len(trades),"summaries":summary,"trades":trades})
    return {
        "project":"CRYPTO-RESEARCH-V1","version":"0.9.0",
        "model":"FA_B_EXIT_RULE_COMPARISON",
        "paper_only":True,"orders":False,
        "symbol":symbol,"days":days,
        "split_utc":datetime.fromtimestamp(split/1000,timezone.utc).isoformat(),
        "data_source":"Binance Vision monthly ZIP archives",
        "rules":{
            "fixed":"Exit at 30/45 days",
            "last_3d_negative":"Exit at next daily open when last 3 completed days' cumulative funding is negative",
            "last_7d_negative":"Exit at next daily open when last 7 completed days' cumulative funding is negative",
            "three_negative_days":"Exit at next daily open after 3 consecutive completed days with negative daily funding",
        },
        "results":output,
        "limitations":[
            "Different exit rules produce different subsequent entries; results are not matched-trade comparisons.",
            "Rules selected after inspecting historical diagnostics: severe data snooping risk.",
            "Very small independent trade counts; validation period was previously inspected.",
            "Exit signals only use funding events strictly before daily open, not future funding.",
            "Daily open approximations and daily perp-open funding notional; intraday risk not modeled.",
            "Fees approximate four fills; excludes spread, slippage, financing, margin and liquidation.",
            "Returns are relative to spot entry notional, not total committed capital.",
            "Historical research only; no live trades or proof of profitability."
        ]
    }


# V1.0 - Extended historical test, rules unchanged.
@app.get("/funding/extended-backtest")
async def funding_extended_backtest(
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"] = "BTCUSDT",
    days: int = Query(730, ge=365, le=1095),
    lookback_days: int = Query(7, ge=3, le=14),
    min_past_funding_pct: float = Query(0.05, ge=0, le=1),
):
    import bisect
    DAY, HOUR = 86400000, 3600000
    now = datetime.now(timezone.utc)
    end = int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp()*1000) - DAY
    start = end - days*DAY
    split = start + (days*2//3)*DAY
    spot, perp, events = await _archive_bundle(symbol, start-(lookback_days+3)*DAY, end)
    times = [t for t, _ in events]
    if len(events)<days or len(spot.keys() & perp.keys())<days*.9:
        raise HTTPException(503, "Yeterli arşiv verisi bulunamadı")
    output=[]
    rules = ("fixed", "last_3d_negative", "last_7d_negative", "three_negative_days")
    for hold in (30,45):
        for rule in rules:
            trades=[]
            entry=start
            while entry+hold*DAY <= end:
                if entry not in spot or entry not in perp:
                    entry+=DAY; continue
                prior=events[bisect.bisect_left(times,entry-lookback_days*DAY):
                             bisect.bisect_left(times,entry)]
                prior_gaps=[entry-lookback_days*DAY]+[t for t,_ in prior]+[entry]
                if len(prior)<lookback_days or any(b-a>36*HOUR for a,b in zip(prior_gaps,prior_gaps[1:])):
                    entry+=DAY; continue
                trailing=100*sum(r for _,r in prior)
                if trailing<min_past_funding_pct:
                    entry+=DAY; continue
                if entry<split<entry+hold*DAY:
                    entry=split; continue
                s0,f0=spot[entry],perp[entry]
                if min(s0,f0)<=0:
                    entry+=DAY; continue
                exit_t=None
                reason="fixed_hold"
                streak=0
                for d in range(1,hold+1):
                    day=entry+d*DAY
                    if day not in spot or day not in perp:
                        break
                    # Funding events strictly earlier than today's daily open are known.
                    today_events=events[bisect.bisect_left(times,day-DAY):
                                        bisect.bisect_left(times,day)]
                    if len(today_events)<1:
                        break
                    today_sum=sum(rate for _,rate in today_events)
                    streak=streak+1 if today_sum<0 else 0
                    if rule=="last_3d_negative" and d>=3:
                        recent=events[bisect.bisect_left(times,day-3*DAY):
                                      bisect.bisect_left(times,day)]
                        trigger=sum(rate for _,rate in recent)<0
                    elif rule=="last_7d_negative" and d>=7:
                        recent=events[bisect.bisect_left(times,day-7*DAY):
                                      bisect.bisect_left(times,day)]
                        trigger=sum(rate for _,rate in recent)<0
                    elif rule=="three_negative_days":
                        trigger=streak>=3
                    else:
                        trigger=False
                    if trigger:
                        exit_t=day
                        reason=rule
                        break
                    if d==hold:
                        exit_t=day
                if exit_t is None:
                    entry+=DAY; continue
                actual=events[bisect.bisect_right(times,entry):
                              bisect.bisect_left(times,exit_t)]
                gaps=[entry]+[t for t,_ in actual]+[exit_t]
                if len(actual)<(exit_t-entry)//DAY or any(b-a>36*HOUR for a,b in zip(gaps,gaps[1:])):
                    entry+=DAY; continue
                if any((t//DAY)*DAY not in perp for t,_ in actual):
                    entry+=DAY; continue
                funding=sum(100*r*perp[(t//DAY)*DAY]/s0 for t,r in actual)
                s1,f1=spot[exit_t],perp[exit_t]
                basis=100*((s1-s0)+(f0-f1))/s0
                gross=funding+basis
                four=1+s1/s0+f0/s0+f1/s0
                trades.append({
                    "entry_utc":datetime.fromtimestamp(entry/1000,timezone.utc).isoformat(),
                    "exit_utc":datetime.fromtimestamp(exit_t/1000,timezone.utc).isoformat(),
                    "partition":"development" if exit_t<=split else "validation",
                    "days_held":(exit_t-entry)//DAY,
                    "exit_reason":reason,
                    "gross_pct":round(gross,6),
                    "net_low_pct":round(gross-0.02*four,6),
                    "net_medium_pct":round(gross-0.04*four,6),
                    "net_baseline_pct":round(gross-0.1*four,6),
                })
                entry=exit_t+DAY
            summary=[]
            for partition in ("development","validation"):
                subset=[t for t in trades if t["partition"]==partition]
                row={"partition":partition,"trades":len(subset),
                     "early_exits":sum(t["exit_reason"]!="fixed_hold" for t in subset)}
                for cost in ("low","medium","baseline"):
                    vals=[t[f"net_{cost}_pct"] for t in subset]
                    row[f"mean_net_{cost}_pct"]=round(sum(vals)/len(vals),6) if vals else None
                    row[f"positive_{cost}_pct"]=round(100*sum(v>0 for v in vals)/len(vals),2) if vals else None
                    eq=1.0
                    for v in vals: eq*=1+v/100
                    row[f"illustrative_compounded_{cost}_pct"]=round(100*(eq-1),6) if vals else None
                summary.append(row)
            output.append({"hold_days":hold,"exit_rule":rule,
                           "total_trades":len(trades),"summaries":summary,"trades":trades})
    return {
        "project":"CRYPTO-RESEARCH-V1","version":"1.0.0",
        "model":"FA_B_EXTENDED_HISTORICAL_BACKTEST",
        "paper_only":True,"orders":False,
        "symbol":symbol,"days":days,
        "split_utc":datetime.fromtimestamp(split/1000,timezone.utc).isoformat(),
        "data_source":"Binance Vision monthly ZIP archives",
        "rules":{
            "fixed":"Exit at 30/45 days",
            "last_3d_negative":"Exit at next daily open when last 3 completed days' cumulative funding is negative",
            "last_7d_negative":"Exit at next daily open when last 7 completed days' cumulative funding is negative",
            "three_negative_days":"Exit at next daily open after 3 consecutive completed days with negative daily funding",
        },
        "results":output,
        "limitations":[
            "Different exit rules produce different subsequent entries; results are not matched-trade comparisons.",
            "Historical archive availability may be incomplete; missing months cause a clear error.",
            "Rules selected after inspecting historical diagnostics: severe data snooping risk.",
            "Very small independent trade counts; validation period was previously inspected.",
            "Exit signals only use funding events strictly before daily open, not future funding.",
            "Daily open approximations and daily perp-open funding notional; intraday risk not modeled.",
            "Fees approximate four fills; excludes spread, slippage, financing, margin and liquidation.",
            "Returns are relative to spot entry notional, not total committed capital.",
            "Historical research only; no live trades or proof of profitability."
        ]
    }


# V1.1: year-by-year attribution for the two preselected 45-day research candidates.
@app.get("/funding/yearly-risk")
async def funding_yearly_risk(
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"] = "BTCUSDT",
    days: int = Query(1095, ge=365, le=1095),
):
    import bisect
    from collections import defaultdict
    DAY = 86400000
    # Reuse V1.0 entry/exit engine, preserving all historical rules.
    original = await funding_extended_backtest(symbol=symbol, days=days,
                                                lookback_days=7, min_past_funding_pct=0.05)
    selected = [r for r in original["results"] if r["hold_days"] == 45
                and r["exit_rule"] in ("fixed", "last_3d_negative")]
    if not selected:
        raise HTTPException(503, "V1.0 strateji verisi bulunamadı")
    all_trades = [t for r in selected for t in r["trades"]]
    if not all_trades:
        raise HTTPException(503, "Analiz edilecek işlem bulunamadı")
    first = min(int(datetime.fromisoformat(t["entry_utc"]).timestamp()*1000) for t in all_trades)
    last = max(int(datetime.fromisoformat(t["exit_utc"]).timestamp()*1000) for t in all_trades)
    spot, perp, events = await _archive_bundle(symbol, first-DAY, last+DAY)
    times = [t for t,_ in events]
    output = []
    for strategy in selected:
        by_year = defaultdict(list)
        detailed = []
        for t in strategy["trades"]:
            a = int(datetime.fromisoformat(t["entry_utc"]).timestamp()*1000)
            b = int(datetime.fromisoformat(t["exit_utc"]).timestamp()*1000)
            s0, f0, s1, f1 = spot.get(a), perp.get(a), spot.get(b), perp.get(b)
            if not all(x is not None and x > 0 for x in (s0, f0, s1, f1)):
                raise HTTPException(503, "İşlem fiyat arşivi eksik")
            actual = events[bisect.bisect_right(times,a):bisect.bisect_left(times,b)]
            if any((ts//DAY)*DAY not in perp for ts,_ in actual):
                raise HTTPException(503, "Funding fiyat arşivi eksik")
            funding_pct = 100*sum(rate*perp[(ts//DAY)*DAY]/s0 for ts,rate in actual)
            basis_pct = 100*((s1-s0)+(f0-f1))/s0
            gross_pct = funding_pct+basis_pct
            fee_pct = .1*(1+s1/s0+f0/s0+f1/s0)
            # Daily synthetic gross marks; NOT margin or liquidation risk.
            marks=[]
            for d in range(a+DAY,b+DAY,DAY):
                if d not in spot or d not in perp:
                    continue
                earned=events[bisect.bisect_right(times,a):bisect.bisect_left(times,d)]
                fp=100*sum(rate*perp[(ts//DAY)*DAY]/s0 for ts,rate in earned)
                bp=100*((spot[d]-s0)+(f0-perp[d]))/s0
                marks.append(fp+bp)
            worst_mark=min(marks) if marks else None
            peak=0.0
            max_dd=0.0
            for mark in marks:
                peak=max(peak,mark)
                max_dd=max(max_dd,peak-mark)
            detail={
                "entry_utc":t["entry_utc"],"exit_utc":t["exit_utc"],
                "partition":t["partition"],"days_held":t["days_held"],
                "exit_reason":t["exit_reason"],
                "funding_pct":round(funding_pct,6),
                "basis_pct":round(basis_pct,6),
                "gross_pct":round(gross_pct,6),
                "four_fill_fee_baseline_pct":round(fee_pct,6),
                "net_baseline_pct":round(gross_pct-fee_pct,6),
                "worst_daily_mark_gross_pct":round(worst_mark,6) if worst_mark is not None else None,
                "max_daily_peak_to_trough_pct_points":round(max_dd,6)
            }
            detailed.append(detail)
            by_year[datetime.fromisoformat(t["exit_utc"]).year].append(detail)
        years=[]
        for year, subset in sorted(by_year.items()):
            n=len(subset)
            def mean(field): return round(sum(t[field] for t in subset)/n,6)
            compounded=1
            for t in subset: compounded*=1+t["net_baseline_pct"]/100
            years.append({
                "exit_year":year,"trades":n,
                "mean_funding_pct":mean("funding_pct"),
                "mean_basis_pct":mean("basis_pct"),
                "mean_fee_pct":mean("four_fill_fee_baseline_pct"),
                "mean_net_pct":mean("net_baseline_pct"),
                "positive_net_pct":round(100*sum(t["net_baseline_pct"]>0 for t in subset)/n,2),
                "illustrative_compounded_net_pct":round(100*(compounded-1),6),
                "worst_daily_mark_gross_pct":min(t["worst_daily_mark_gross_pct"] for t in subset if t["worst_daily_mark_gross_pct"] is not None),
            })
        output.append({"hold_days":45,"exit_rule":strategy["exit_rule"],
                       "total_trades":len(detailed),"yearly":years,"trades":detailed})
    return {
        "project":"CRYPTO-RESEARCH-V1","version":"1.1.0",
        "model":"FA_B_YEARLY_ATTRIBUTION_RISK","paper_only":True,"orders":False,
        "symbol":symbol,"days":days,
        "baseline_fee_model":"10 bps per fill x 4 fills, price-weighted; financing excluded",
        "results":output,
        "limitations":[
            "Calendar year is based on exit date; partial 2023 and 2026 years.",
            "The two strategies may enter at different dates, so not a matched-trade comparison.",
            "Synthetic daily spot-long/perp-short marks are not executable liquidation values.",
            "Intraday extremes, collateral requirements, financing, borrow rates, margin and liquidation are NOT modeled.",
            "No spread or slippage included beyond illustrative fee scenario.",
            "Returns relative to spot notional, not actual deployed capital.",
            "Funding accrual approximates perp notional using daily open.",
            "Previously inspected history is not independent out-of-sample validation.",
            "Research paper only; no trading or orders."
        ]
    }

# V1.2: cost sensitivity and illustrative collateral stress, not an exchange margin simulator.
@app.get('/funding/cost-collateral-analysis')
async def funding_cost_collateral_analysis(
    symbol: Literal['BTCUSDT','ETHUSDT','SOLUSDT'] = 'BTCUSDT',
    days: int = Query(1095,ge=365,le=1095),
    financing_annual_pct: float = Query(0.0,ge=0,le=30),
    collateral_fraction_of_spot_notional: float = Query(1.0,ge=0.1,le=3.0),
):
    from collections import defaultdict
    prior = await funding_yearly_risk(symbol=symbol,days=days)
    scenarios = [
        ('low',0.02,0.0),
        ('medium',0.04,0.0),
        ('baseline',0.10,0.0),
        ('baseline_plus_2bps_slippage_each_fill',0.10,0.02),
        ('baseline_plus_5bps_slippage_each_fill',0.10,0.05),
        ('high_fee_plus_5bps_slippage_each_fill',0.15,0.05),
    ]
    results=[]
    for strategy in prior['results']:
        rows=[]
        for scenario,fee_per_fill_pct,slippage_per_fill_pct in scenarios:
            trades=[]
            for trade in strategy['trades']:
                # Same four price-weighted notionals used by the existing V1.1 fee baseline.
                four_notional_factor = trade['four_fill_fee_baseline_pct']/0.1
                execution_cost=(fee_per_fill_pct+slippage_per_fill_pct)*four_notional_factor
                financing_cost=(financing_annual_pct*trade['days_held']/365.0)
                net=trade['gross_pct']-execution_cost-financing_cost
                # Simplified cash allocation assumes both legs are fully collateralized
                # with cash equal to (1 + selected collateral fraction) x spot notional.
                deployed_capital_factor=1.0+collateral_fraction_of_spot_notional
                trades.append({
                    'exit_year':int(trade['exit_utc'][:4]),
                    'partition':trade['partition'],
                    'net_pct_spot_notional':net,
                    'net_pct_allocated_capital':net/deployed_capital_factor,
                    'funding_pct':trade['funding_pct'],
                    'execution_cost_pct':execution_cost,
                    'financing_cost_pct':financing_cost,
                    'days_held':trade['days_held'],
                })
            for group_name,group in [('all',trades),('development',[t for t in trades if t['partition']=='development']),('validation',[t for t in trades if t['partition']=='validation'])]:
                if not group:continue
                rows.append({
                    'scenario':scenario,'period':group_name,'trades':len(group),
                    'mean_net_pct_spot_notional':round(sum(t['net_pct_spot_notional'] for t in group)/len(group),6),
                    'mean_net_pct_allocated_capital':round(sum(t['net_pct_allocated_capital'] for t in group)/len(group),6),
                    'win_rate_pct':round(100*sum(t['net_pct_spot_notional']>0 for t in group)/len(group),2),
                    'mean_execution_cost_pct':round(sum(t['execution_cost_pct'] for t in group)/len(group),6),
                    'mean_financing_cost_pct':round(sum(t['financing_cost_pct'] for t in group)/len(group),6),
                })
            for year in sorted({t['exit_year'] for t in trades}):
                subset=[t for t in trades if t['exit_year']==year]
                rows.append({
                    'scenario':scenario,'period':str(year),'trades':len(subset),
                    'mean_net_pct_spot_notional':round(sum(t['net_pct_spot_notional'] for t in subset)/len(subset),6),
                    'mean_net_pct_allocated_capital':round(sum(t['net_pct_allocated_capital'] for t in subset)/len(subset),6),
                    'win_rate_pct':round(100*sum(t['net_pct_spot_notional']>0 for t in subset)/len(subset),2),
                })
        results.append({'exit_rule':strategy['exit_rule'],'hold_days':45,'total_trades':strategy['total_trades'],'scenarios':rows})
    return {
        'project':'CRYPTO-RESEARCH-V1','version':'1.2.0',
        'model':'FA_B_COST_COLLATERAL_SENSITIVITY','paper_only':True,'orders':False,
        'symbol':symbol,'days':days,'financing_annual_pct':financing_annual_pct,
        'collateral_fraction_of_spot_notional':collateral_fraction_of_spot_notional,
        'assumed_allocated_capital_multiple_of_spot_notional':1+collateral_fraction_of_spot_notional,
        'scenarios':scenarios,'results':results,
        'limitations':[
            'Funding values are from V1.1, which approximates funding notional with daily perp opens; NOT settlement-time pricing.',
            'Fees and slippage are hypothetical and may differ from actual venue/tier/market conditions.',
            'Financing cost is a simple annualized input on spot notional, not a verified borrow quote.',
            'Collateral fraction is only a user-selected capital-allocation denominator, NOT an exchange maintenance-margin or liquidation model.',
            'No intraday high/low, liquidation engine, borrowing availability, ADL, margin calls or portfolio cross-margin.',
            'Prior development/validation periods have been inspected; no independent out-of-sample proof.',
            'Research only; no live trading, no orders.'
        ]
    }


# V1.3 — historical funding-event integrity and execution-cost audit.
# Uses archived funding timestamps/rates, never invents settlement prices.
@app.get("/funding/v13-audit")
async def funding_v13_audit(
    symbol: Literal["BTCUSDT", "ETHUSDT", "SOLUSDT"] = "BTCUSDT",
    days: int = Query(1095, ge=365, le=1095),
    fee_per_fill_bps: float = Query(10.0, ge=0, le=50),
    slippage_per_fill_bps: float = Query(2.0, ge=0, le=50),
    financing_annual_pct: float = Query(0.0, ge=0, le=30),
):
    import bisect
    from collections import defaultdict
    DAY = 86400000
    prior = await funding_yearly_risk(symbol=symbol, days=days)
    trades = [t for strategy in prior["results"] for t in strategy["trades"]]
    if not trades:
        raise HTTPException(503, "No historical trades available")
    def millis(value):
        return int(datetime.fromisoformat(value).timestamp()*1000)
    earliest = min(millis(t["entry_utc"]) for t in trades)
    latest = max(millis(t["exit_utc"]) for t in trades)
    spot, perp, events = await _archive_bundle(symbol, earliest-DAY, latest+DAY)
    times = [t for t, _ in events]
    yearly = defaultdict(list)
    gaps = []
    unusual_timestamps = []
    duplicate_count = 0
    seen = set()
    for ts, rate in events:
        if ts in seen:
            duplicate_count += 1
        seen.add(ts)
        yearly[datetime.fromtimestamp(ts/1000, timezone.utc).year].append(rate)
        if ts % (3600*1000) != 0:
            unusual_timestamps.append(datetime.fromtimestamp(ts/1000, timezone.utc).isoformat())
    for (t0,_),(t1,_) in zip(events,events[1:]):
        hours=(t1-t0)/3600000
        if hours>25:
            gaps.append({"from_utc":datetime.fromtimestamp(t0/1000,timezone.utc).isoformat(),
                         "to_utc":datetime.fromtimestamp(t1/1000,timezone.utc).isoformat(),
                         "hours":round(hours,2)})
    strategies=[]
    for strategy in prior["results"]:
        enriched=[]
        for t in strategy["trades"]:
            a,b=millis(t["entry_utc"]),millis(t["exit_utc"])
            event_slice=events[bisect.bisect_right(times,a):bisect.bisect_left(times,b)]
            # Event-rate sum is exact for the archive; funding P&L is not, without
            # settlement-time perp mark and position sizing history.
            raw_funding_rate_sum_pct=100*sum(rate for _,rate in event_slice)
            total_exec_cost_pct=(fee_per_fill_bps+slippage_per_fill_bps)/10.0 * t["four_fill_fee_baseline_pct"]
            financing_cost_pct=financing_annual_pct*t["days_held"]/365.0
            net=t["gross_pct"]-total_exec_cost_pct-financing_cost_pct
            enriched.append({
                "entry_utc":t["entry_utc"],"exit_utc":t["exit_utc"],
                "partition":t["partition"],"exit_year":int(t["exit_utc"][:4]),
                "days_held":t["days_held"],"funding_events":len(event_slice),
                "raw_funding_rate_sum_pct":round(raw_funding_rate_sum_pct,6),
                "daily_open_estimated_funding_pnl_pct":t["funding_pct"],
                "estimated_basis_pct":t["basis_pct"],
                "execution_cost_pct":round(total_exec_cost_pct,6),
                "financing_cost_pct":round(financing_cost_pct,6),
                "scenario_net_pct_spot_notional":round(net,6),
                "settlement_price_verified":False
            })
        summaries=[]
        for group_name,subset in [
            ("development",[t for t in enriched if t["partition"]=="development"]),
            ("validation",[t for t in enriched if t["partition"]=="validation"])
        ] + [(str(y),[t for t in enriched if t["exit_year"]==y])
             for y in sorted({t["exit_year"] for t in enriched})]:
            if not subset:
                continue
            summaries.append({
                "period":group_name,"trades":len(subset),
                "mean_net_pct_spot_notional":round(sum(t["scenario_net_pct_spot_notional"] for t in subset)/len(subset),6),
                "win_rate_pct":round(100*sum(t["scenario_net_pct_spot_notional"]>0 for t in subset)/len(subset),2),
                "mean_raw_funding_rate_sum_pct":round(sum(t["raw_funding_rate_sum_pct"] for t in subset)/len(subset),6),
                "mean_estimated_funding_pnl_pct":round(sum(t["daily_open_estimated_funding_pnl_pct"] for t in subset)/len(subset),6),
                "mean_execution_cost_pct":round(sum(t["execution_cost_pct"] for t in subset)/len(subset),6)
            })
        strategies.append({"exit_rule":strategy["exit_rule"],"hold_days":45,
                           "summaries":summaries,"trades":enriched})
    return {
        "project":"CRYPTO-RESEARCH-V1","version":"1.3.0",
        "model":"FA_B_FUNDING_DATA_AND_COST_AUDIT","paper_only":True,"orders":False,
        "symbol":symbol,"days":days,
        "inputs":{"fee_per_fill_bps":fee_per_fill_bps,
                  "slippage_per_fill_bps":slippage_per_fill_bps,
                  "financing_annual_pct":financing_annual_pct},
        "archive_quality":{
            "funding_event_count":len(events),
            "unique_funding_timestamps":len(seen),
            "duplicate_timestamps":duplicate_count,
            "non_hour_aligned_timestamps_sample":unusual_timestamps[:10],
            "gaps_over_25h_count":len(gaps),
            "gaps_over_25h_sample":gaps[:12],
            "funding_by_year":[{"year":y,"events":len(rates),
                                "positive_pct":round(100*sum(r>0 for r in rates)/len(rates),2),
                                "negative_pct":round(100*sum(r<0 for r in rates)/len(rates),2),
                                "sum_raw_rates_pct":round(100*sum(rates),6)}
                               for y,rates in sorted(yearly.items())],
            "daily_spot_open_rows":len(spot),"daily_perp_open_rows":len(perp),
            "settlement_time_perp_prices_available":False
        },
        "strategies":strategies,
        "limitations":[
            "Funding event timestamps and rates are read from Binance Vision monthly archives.",
            "Settlement-time perp mark prices are NOT present in the daily-open archive.",
            "Raw funding rate sums are exact sums of archived rates, NOT exact funding cashflow.",
            "Funding cashflow and basis P&L retain V1.1 daily-open approximations.",
            "Fees, slippage, and financing are user-selected assumptions, not verified exchange quotes.",
            "No margin, liquidation, intraday exposure or funding-payment cash ledger.",
            "Archive quality checks do not establish exchange-level reconciliation.",
            "Development and validation periods were previously inspected; not independent proof.",
            "Paper research only. No orders or trading."
        ]
    }


# V1.4 — bundled independent diagnostics. No trading or order functions.
@app.get("/funding/v14-research")
async def funding_v14_research(
    symbol: Literal["BTCUSDT","ETHUSDT","SOLUSDT"] = "BTCUSDT",
    days: int = Query(1095, ge=365, le=1095),
    fee_per_fill_bps: float = Query(10.0, ge=0, le=50),
    slippage_per_fill_bps: float = Query(2.0, ge=0, le=50),
    financing_annual_pct: float = Query(0.0, ge=0, le=30),
):
    """Funding regime, data completeness, holding windows, cost break-even.

    Important: only archived funding RATES are exact; cashflows use daily-open
    approximations. We intentionally do not label them settlement-verified.
    """
    from collections import Counter, defaultdict
    import statistics
    DAY = 86400000
    audit = await funding_v13_audit(
        symbol=symbol, days=days, fee_per_fill_bps=fee_per_fill_bps,
        slippage_per_fill_bps=slippage_per_fill_bps,
        financing_annual_pct=financing_annual_pct)
    fixed = next(x for x in audit["strategies"] if x["exit_rule"]=="fixed")
    early = next(x for x in audit["strategies"] if x["exit_rule"]=="last_3d_negative")
    trades = fixed["trades"] + early["trades"]
    a = min(int(datetime.fromisoformat(t["entry_utc"]).timestamp()*1000) for t in trades)
    b = max(int(datetime.fromisoformat(t["exit_utc"]).timestamp()*1000) for t in trades)
    spot, perp, events = await _archive_bundle(symbol,a-45*DAY,b+DAY)
    # Monthly rate sums and negative-event frequency; partial months explicitly flagged.
    months=defaultdict(list)
    for ts,r in events:
        key=datetime.fromtimestamp(ts/1000,timezone.utc).strftime("%Y-%m")
        months[key].append((ts,r))
    monthly=[]
    for month, values in sorted(months.items()):
        monthly.append({
            "month":month,"events":len(values),
            "raw_funding_rate_sum_pct":round(100*sum(r for _,r in values),6),
            "negative_event_pct":round(100*sum(r<0 for _,r in values)/len(values),2),
            "partial_month_possible":month in (min(months),max(months))
        })
    # Interval distribution detects 8h / 4h / 1h schedule changes and short gaps.
    interval_counts=Counter()
    for (t0,_),(t1,_) in zip(events,events[1:]):
        hours=(t1-t0)/3600000
        interval_counts[str(round(hours,2))]+=1
    # Prospectively defined trailing-rate signals, calculated ONLY from prior events.
    # Descriptive forward sums of rates (not P&L) at non-overlapping monthly anchors.
    event_times=[t for t,_ in events]
    import bisect
    regimes=[]
    for month in sorted(months):
        y,m=map(int,month.split("-"))
        anchor=int(datetime(y,m,1,tzinfo=timezone.utc).timestamp()*1000)
        if anchor-30*DAY<a-45*DAY or anchor+30*DAY>b:
            continue
        idx=bisect.bisect_left(event_times,anchor)
        prior7=[r for t,r in events[bisect.bisect_left(event_times,anchor-7*DAY):idx]]
        prior30=[r for t,r in events[bisect.bisect_left(event_times,anchor-30*DAY):idx]]
        forward30=[r for t,r in events[idx:bisect.bisect_left(event_times,anchor+30*DAY)]]
        if len(prior7)<14 or len(prior30)<60 or len(forward30)<60:
            continue
        regimes.append({
            "month":month,"prior_7d_raw_rate_sum_pct":round(100*sum(prior7),6),
            "prior_30d_raw_rate_sum_pct":round(100*sum(prior30),6),
            "next_30d_raw_rate_sum_pct":round(100*sum(forward30),6),
            "prior_7d_positive":sum(prior7)>0,
            "next_30d_positive":sum(forward30)>0
        })
    regime_summary=[]
    for label, subset in (
        ("prior_7d_positive",[r for r in regimes if r["prior_7d_positive"]]),
        ("prior_7d_nonpositive",[r for r in regimes if not r["prior_7d_positive"]])
    ):
        regime_summary.append({
            "group":label,"months":len(subset),
            "mean_next_30d_raw_rate_sum_pct":round(statistics.mean(
                r["next_30d_raw_rate_sum_pct"] for r in subset),6) if subset else None,
            "next_30d_positive_pct":round(100*sum(
                r["next_30d_positive"] for r in subset)/len(subset),2) if subset else None
        })
    # Cost sensitivity on identical entry/exit trades, without reoptimizing signals.
    sensitivities=[]
    for name, f, slip, financing in (
        ("low_cost",2,0,0),
        ("medium_cost",4,1,0),
        ("baseline",10,0,0),
        ("baseline_slip2",10,2,0),
        ("baseline_slip5",10,5,0),
        ("baseline_slip2_financing5pct",10,2,5)
    ):
        scenario={"scenario":name,"fee_per_fill_bps":f,
                  "slippage_per_fill_bps":slip,"financing_annual_pct":financing}
        for strategy in (fixed,early):
            subset=[t for t in strategy["trades"] if t["partition"]=="validation"]
            # Recover gross = V1.3 net + selected cost + selected financing.
            vals=[]
            break_even=[]
            for t in subset:
                baseline=t["execution_cost_pct"]/(fee_per_fill_bps+slippage_per_fill_bps) if (fee_per_fill_bps+slippage_per_fill_bps)>0 else None
                # When user supplied zero fees, infer baseline 10bps/fill from
                # underlying trade prices, not a division by zero.
                if baseline is None:
                    # Approximate 4 fills with original weighted notional factor.
                    baseline=0.04
                gross=t["scenario_net_pct_spot_notional"]+t["execution_cost_pct"]+t["financing_cost_pct"]
                fin=financing*t["days_held"]/365
                vals.append(gross-baseline*(f+slip)-fin)
                if baseline>0:
                    break_even.append(max(0,(gross-fin)/baseline))
            key="fixed" if strategy is fixed else "last_3d_negative"
            scenario[key]={
                "validation_trades":len(vals),
                "mean_net_pct_spot_notional":round(statistics.mean(vals),6) if vals else None,
                "win_rate_pct":round(100*sum(v>0 for v in vals)/len(vals),2) if vals else None,
                "median_break_even_combined_bps_per_fill":round(statistics.median(break_even),3) if break_even else None
            }
        sensitivities.append(scenario)
    # Duration: preexisting entry/exit logic, compare descriptive results only.
    hold_groups=[]
    for strat in (fixed,early):
        for label,subset in (
            ("up_to_14d",[t for t in strat["trades"] if t["days_held"]<=14]),
            ("15_to_29d",[t for t in strat["trades"] if 15<=t["days_held"]<=29]),
            ("30_to_44d",[t for t in strat["trades"] if 30<=t["days_held"]<=44]),
            ("45d",[t for t in strat["trades"] if t["days_held"]==45])
        ):
            hold_groups.append({
                "strategy":strat["exit_rule"],"duration":label,
                "trades":len(subset),
                "mean_net_pct_spot_notional":round(statistics.mean(
                    t["scenario_net_pct_spot_notional"] for t in subset),6) if subset else None,
                "validation_trades":sum(t["partition"]=="validation" for t in subset)
            })
    return {
        "project":"CRYPTO-RESEARCH-V1","version":"1.4.0",
        "model":"FUNDING_MULTISTUDY_DIAGNOSTIC","paper_only":True,"orders":False,
        "symbol":symbol,"days":days,
        "studies":{
            "archive_quality":audit["archive_quality"],
            "funding_intervals_hours":dict(interval_counts.most_common(12)),
            "monthly_funding_rates":monthly,
            "forward_rate_regimes":{
                "definition":"Monthly anchor; preceding 7d vs next 30d; raw rates only",
                "summary":regime_summary,"observations":regimes},
            "validation_cost_sensitivity":sensitivities,
            "observed_trade_duration":hold_groups,
            "strategy_summaries":[{"exit_rule":x["exit_rule"],"summaries":x["summaries"]}
                                  for x in audit["strategies"]]
        },
        "limitations":[
            "No exact settlement-time perp mark prices: funding cashflow NOT verified.",
            "Archive rate sum is not a realized portfolio return.",
            "Funding interval distribution does not by itself prove completeness.",
            "Regime results describe rates, not executable spot/perp arbitrage P&L.",
            "Duration groups are observational, not a causal hold-period test.",
            "Sensitivity uses assumed fees, slippage and financing, not verified user fees.",
            "Financing uses simple annualized approximation on spot notional.",
            "Break-even values are per-fill combined fee+slippage assumptions.",
            "The analyzed periods have already been inspected; not fresh out-of-sample.",
            "No intraday liquidation, basis intraday extremes, margin or ADL modeling.",
            "Research only: no live orders."
        ]
    }


# V1.5: matched-entry holding periods, past-only entry filters, transparent proxy audit.
@app.get("/funding/v15-comparison")
async def funding_v15_comparison(
    symbol: Literal["BTCUSDT","ETHUSDT","SOLUSDT"]="BTCUSDT",
    days: int=Query(1095,ge=365,le=1095),
    fee_per_fill_bps: float=Query(10.0,ge=0,le=50),
    slippage_per_fill_bps: float=Query(2.0,ge=0,le=50),
    financing_annual_pct: float=Query(0.0,ge=0,le=30),
):
    import bisect
    import statistics
    from collections import defaultdict
    DAY=86400000
    now=datetime.now(timezone.utc)
    # Use only completed monthly archives, as in the older endpoints.
    first_of_month=datetime(now.year,now.month,1,tzinfo=timezone.utc)
    end_ms=int(first_of_month.timestamp()*1000)-DAY
    start_ms=end_ms-days*DAY
    warmup_ms=start_ms-32*DAY
    spot,perp,events=await _archive_bundle(symbol,warmup_ms,end_ms)
    event_times=[ts for ts,_ in events]
    event_rates=[r for _,r in events]
    prefix=[0.0]
    for rate in event_rates:
        prefix.append(prefix[-1]+rate)
    def rates_sum(lo,hi):
        i=bisect.bisect_left(event_times,lo)
        j=bisect.bisect_left(event_times,hi)
        return prefix[j]-prefix[i],j-i
    # First monthly UTC open with a complete 30-day historical signal and
    # sufficient future room for ALL four horizons. Same entry across horizons.
    anchors=[]
    for t in sorted(set(spot)&set(perp)):
        dt=datetime.fromtimestamp(t/1000,timezone.utc)
        if dt.day!=1 or t<start_ms or t+45*DAY>end_ms:
            continue
        if t-30*DAY<warmup_ms:
            continue
        if any(t+h*DAY not in spot or t+h*DAY not in perp for h in (7,15,30,45)):
            continue
        prior7,n7=rates_sum(t-7*DAY,t)
        prior30,n30=rates_sum(t-30*DAY,t)
        if n7<14 or n30<60:
            continue
        anchors.append((t,prior7*100,prior30*100))
    if not anchors:
        raise HTTPException(503,detail="V1.5 için eşleşen aylık giriş tarihleri bulunamadı")
    # Predeclared thresholds. Descriptive, NOT optimization or independent validation.
    filters={
        "all_monthly_entries":lambda a,b:True,
        "prior_7d_positive":lambda a,b:a>0,
        "prior_7d_at_least_0_05pct":lambda a,b:a>=0.05,
        "prior_30d_at_least_0_30pct":lambda a,b:b>=0.30,
        "both_7d_positive_and_30d_0_30pct":lambda a,b:a>0 and b>=0.30,
    }
    rows=[]
    for t,prior7,prior30 in anchors:
        s0,f0=spot[t],perp[t]
        if min(s0,f0)<=0:
            continue
        for horizon in (7,15,30,45):
            exit_t=t+horizon*DAY
            s1,f1=spot[exit_t],perp[exit_t]
            if min(s1,f1)<=0:
                continue
            # Cashflow proxy uses daily perp opens, NOT settlement-time mark.
            i=bisect.bisect_right(event_times,t)
            j=bisect.bisect_left(event_times,exit_t)
            chosen=events[i:j]
            missing=sum(1 for ts,_ in chosen if (ts//DAY)*DAY not in perp)
            if missing:
                raise HTTPException(503,detail="V1.5 funding için günlük fiyat eksik")
            funding_proxy_pct=100*sum(r*perp[(ts//DAY)*DAY]/s0 for ts,r in chosen)
            raw_sum_pct=100*sum(r for _,r in chosen)
            basis_pct=100*((s1-s0)+(f0-f1))/s0
            gross_pct=funding_proxy_pct+basis_pct
            # 4 fills: spot buy+sell and perp short+cover; each fee is
            # charged on actual corresponding notional, relative to spot entry.
            fill_notional_ratio=1+s1/s0+f0/s0+f1/s0
            exec_pct=(fee_per_fill_bps+slippage_per_fill_bps)/100*fill_notional_ratio
            fin_pct=financing_annual_pct*horizon/365
            rows.append({
                "entry_utc":datetime.fromtimestamp(t/1000,timezone.utc).strftime("%Y-%m-%d"),
                "entry_year":datetime.fromtimestamp(t/1000,timezone.utc).year,
                "horizon_days":horizon,
                "prior_7d_raw_rate_pct":round(prior7,6),
                "prior_30d_raw_rate_pct":round(prior30,6),
                "events":len(chosen),
                "raw_rate_sum_pct":round(raw_sum_pct,6),
                "funding_daily_open_proxy_pct":round(funding_proxy_pct,6),
                "basis_pct":round(basis_pct,6),
                "execution_cost_pct":round(exec_pct,6),
                "financing_cost_pct":round(fin_pct,6),
                "net_proxy_pct":round(gross_pct-exec_pct-fin_pct,6),
                "settlement_mark_verified":False
            })
    comparisons=[]
    for name,rule in filters.items():
        for h in (7,15,30,45):
            subset=[r for r in rows if r["horizon_days"]==h and
                    rule(r["prior_7d_raw_rate_pct"],r["prior_30d_raw_rate_pct"])]
            values=[r["net_proxy_pct"] for r in subset]
            comparisons.append({
                "filter":name,"horizon_days":h,"trades":len(values),
                "mean_net_proxy_pct":round(statistics.mean(values),6) if values else None,
                "median_net_proxy_pct":round(statistics.median(values),6) if values else None,
                "win_rate_pct":round(100*sum(v>0 for v in values)/len(values),2) if values else None,
                "worst_net_proxy_pct":round(min(values),6) if values else None,
                "by_entry_year":[{
                    "year":year,"trades":len(ys),
                    "mean_net_proxy_pct":round(statistics.mean(ys),6)
                } for year in sorted({r["entry_year"] for r in subset})
                  for ys in [[r["net_proxy_pct"] for r in subset if r["entry_year"]==year]]]
            })
    # Exact matching of source windows: V1.4 compared different windows.
    source_audit={
        "dataset_window_start_utc":datetime.fromtimestamp(warmup_ms/1000,timezone.utc).isoformat(),
        "dataset_window_end_utc":datetime.fromtimestamp(end_ms/1000,timezone.utc).isoformat(),
        "funding_events_in_window":len(events),
        "funding_intervals":len(events)-1,
        "spot_daily_open_rows":len(spot),
        "perp_daily_open_rows":len(perp),
        "monthly_matched_entry_count":len(anchors),
        "daily_open_prices_only":True,
        "settlement_mark_price_source":"UNAVAILABLE_NOT_VERIFIED",
        "funding_cashflow_reconciled":False
    }
    return {
        "project":"CRYPTO-RESEARCH-V1","version":"1.5.0",
        "model":"MATCHED_HORIZON_PAST_ONLY_FILTER_RESEARCH",
        "paper_only":True,"trading":False,"orders":False,
        "symbol":symbol,"days":days,
        "inputs":{"fee_per_fill_bps":fee_per_fill_bps,
                  "slippage_per_fill_bps":slippage_per_fill_bps,
                  "financing_annual_pct":financing_annual_pct,
                  "horizons_days":[7,15,30,45]},
        "data_audit":source_audit,
        "comparisons":comparisons,
        "monthly_matched_entries":rows,
        "limitations":[
            "Settlement-time mark prices have not been retrieved or verified. Funding uses DAILY OPEN proxy.",
            "Historical funding rate sums are not actual credited funding payments.",
            "All horizons share monthly entry dates; results are correlated, not independent.",
            "Monthly sampling reduces overlap but also yields very few observations.",
            "Filter thresholds were chosen after prior history was inspected; no fresh out-of-sample evidence.",
            "Missing future horizon near dataset end is excluded equally across horizons.",
            "This model assumes equal base-asset quantities for spot and perp short.",
            "Capital denominators are spot-entry notional, not full cash/margin collateral.",
            "No borrowing availability, maintenance margin, liquidation, ADL, intraday basis risk.",
            "Financing and execution fees are assumptions, not observed exchange charges.",
            "This endpoint does not simulate orders and must not be used for live trading."
        ]
    }
