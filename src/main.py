import os
import asyncio
import aiohttp
import logging
from log import configure_logging
import yfinance as yf
import pandas as pd
from render_html import render_html_table


log = logging.getLogger(__name__)


# ---------- Config ----------
APP_TOKEN = os.environ.get("APP_TOKEN")
USER_KEY = os.environ.get("USER_KEY")
PUSH = os.environ.get("PUSH", "false").lower() == "true"
TICKERS = os.environ.get("TICKERS", "MSFT,AAPL,GOOGL").split(",")

FAST, SLOW, SIGNAL = 12, 26, 9
SMA_WINDOW = 30
RSI_PERIOD = 14

TICKERS = ('RHM.VI','HUT,ASPI', 'KIN2.DE', 'DRH.SG', 'CJ6.F', '49V.F', 'BSPA.F', 'RQ0.F', '98W.F', 'NVDA', 'VA3.F', 'TOA.F', 'NMM.F', 'MSFT', 'AAPL', 'AMZN', 'GOOGL', 'IBM', 'INTC', 'META', 'PA2.F', 'YO0.F', 'BTC-EUR', 'ETH-EUR', 'XRP-EUR', 'LTC-EUR', 'ADA-EUR', 'BNB-EUR', 'DOT-EUR', 'SOL-EUR')


# ---------- Indicator Weights ----------
weights = {
    "RSI_entry": 1,
    "RSI_exit": 1,
    "RSI_performance": 1,
    "MACD_entry": 1,
    "MACD_exit": 1,
    "MACD_performance": 1,
    "SMA_entry": 1,
    "SMA_exit": 1,
    "SMA_performance": 1,
    "Bollinger_entry": 1,
    "Bollinger_exit": 1,
    "Bollinger_performance": 1,
    "Volume_entry": 1,
    "Volume_exit": 1,
    "Volume_performance": 1,
    "PVT_entry": 1,
    "PVT_exit": 1,
    "PVT_performance": 1,
}


# ---------- Utility Functions ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    df["EMA_FAST"] = ema(df["Close"], FAST)
    df["EMA_SLOW"] = ema(df["Close"], SLOW)
    df["MACD"] = df["EMA_FAST"] - df["EMA_SLOW"]
    df["MACD_SIGNAL"] = ema(df["MACD"], SIGNAL)
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    df["SMA30"] = df["Close"].rolling(SMA_WINDOW).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["YCP"] = df["Close"].shift(1)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    df["RSI"] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-6)))  # avoid divide by zero

    return df


# ----------  rank from Yahoo ----------
def get_recommendation(symbol: str) -> dict:
    """
    Fetch the latest analyst recommendations for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations
        # catch if statuscode is 404
        if recommendations is None:
            return {}
    except Exception as e:
        print(f"Error fetching recommendations for {symbol}: {e}")
        return {}

    if recommendations is not None and not recommendations.empty:
        recent = recommendations[recommendations["period"] == "0m"].copy()
        if not recent.empty:
            recent["total"] = recent[
                ["strongBuy", "buy", "hold", "sell", "strongSell"]
            ].sum(axis=1)
            recent = recent.drop(columns=["period"])
            result = recent.iloc[0].to_dict()

            # Calculate verdict
            buys = result["strongBuy"] + result["buy"]
            sells = result["sell"] + result["strongSell"]
            holds = result["hold"]

            if result["total"] < 10:
                verdict = "Insufficient data"
            elif buys > sells and buys > holds:
                verdict = "Buy"
            elif sells > buys and sells > holds:
                verdict = "Sell"
            else:
                verdict = "Hold"

            result["verdict"] = verdict
            return result
    return {}


def fetch_history(ticker: str, period="230d", interval="1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(
            period=period, interval=interval, auto_adjust=True, prepost=False
        )
        if df.empty or len(df) < int("".join(filter(str.isdigit, period))):
            return pd.DataFrame()
        df.attrs["ticker"] = ticker
        return df
    except Exception:
        return pd.DataFrame()


# ---------- Signal, Score & Classification ----------
def compute_scores(
    ticker,
    last,
    macd_hist_condition,
    price_trend,
    band_position,
    above_sma,
    volume_condition,
    pvt_trend,
    macd_trend_5,
    weights,
):
    rsi_performance_score = rsi_entry_score = rsi_exit_score = 0
    macd_performance_score = macd_entry_score = macd_exit_score = 0
    sma_performance_score = sma_entry_score = sma_exit_score = 0
    vol_performance_score = vol_entry_score = vol_exit_score = 0
    pvt_performance_score = pvt_entry_score = pvt_exit_score = 0

    rsi = last["RSI"]
    macd = last["MACD"]
    macd_signal = last["MACD_SIGNAL"]
    macd_hist = last["MACD_HIST"]

    log.info(
        f"Computing scores for {ticker} - Price: {last['Close']:.2f}, MACD: {macd:.4f}, Signal: {macd_signal:.4f}, Hist: {macd_hist:.4f}"
    )
    log.info(
        f"Price Trend: {price_trend}, Bollinger: {band_position}, Above SMA30: {above_sma}, Volume: {volume_condition}, PVT Trend: {pvt_trend}"
    )
    log.info(f"MACD Histogram Condition: {macd_hist_condition}")
    log.info(f"RSI: {rsi:.2f}")
    log.info(f"Volume Condition: {volume_condition}, PVT Trend: {pvt_trend}")

    # RSI
    if rsi < 30:
        rsi_entry_score += 2
        rsi_exit_score -= 2
        rsi_performance_score -= 2
    elif rsi < 40:
        rsi_entry_score += 1 * weights["RSI_entry"]
        rsi_exit_score -= 1 * weights["RSI_exit"]
        rsi_performance_score -= 1 * weights["RSI_performance"]
    elif rsi > 80:
        rsi_entry_score -= 2 * weights["RSI_entry"]
        rsi_exit_score += 2 * weights["RSI_exit"]
        rsi_performance_score += 2 * weights["RSI_performance"]
    elif rsi > 70:
        rsi_entry_score -= 1 * weights["RSI_entry"]
        rsi_exit_score += 1 * weights["RSI_exit"]
        rsi_performance_score += 1 * weights["RSI_performance"]
    else:
        rsi_performance_score += 1
        rsi_entry_score += 0
        rsi_exit_score += 0 if 40 <= rsi <= 60 else 0

    log.info(
        f"RSI: {rsi:.2f}, Entry Score: {rsi_entry_score}, Exit Score: {rsi_exit_score}, Performance Score: {rsi_performance_score}"
    )

    # MACD & Histogram
    # Bullish: MACD above signal, histogram rising, both below zero (early reversal)
    if macd > macd_signal and macd_hist_condition == "increasing" and macd < 0:
        macd_entry_score += 3 * weights["MACD_entry"]
        macd_exit_score += 0 * weights["MACD_exit"]
        macd_performance_score += 2 * weights["MACD_performance"]

    # Strong Bullish: MACD above signal, histogram rising, both above zero (trend continuation)
    elif (
        macd > macd_signal
        and macd_hist_condition == "increasing"
        and macd_trend_5 == "increasing"
        and macd > 0
    ):
        macd_entry_score += 4 * weights["MACD_entry"]
        macd_exit_score += 0.5 * weights["MACD_exit"]
        macd_performance_score += 3 * weights["MACD_performance"]

    # Bearish: MACD below signal, histogram falling, both above zero (early reversal down)
    elif macd < macd_signal and macd_hist_condition == "decreasing" and macd > 0:
        macd_entry_score += 0 * weights["MACD_entry"]
        macd_exit_score += 3 * weights["MACD_exit"]
        macd_performance_score += 0 * weights["MACD_performance"]

    # Strong Bearish: MACD below signal, histogram falling, both below zero (trend continuation)
    elif macd < macd_signal and macd_hist_condition == "decreasing" and macd < 0:
        macd_entry_score += 0 * weights["MACD_entry"]
        macd_exit_score += 4 * weights["MACD_exit"]
        macd_performance_score -= 1 * weights["MACD_performance"]

    # Weak/Neutral bullish cross (MACD > signal, histogram decreasing)
    elif (
        macd > macd_signal
        and macd_hist_condition == "decreasing"
        and macd_trend_5 != "increasing"
    ):
        macd_entry_score += 1 * weights["MACD_entry"]
        macd_exit_score += 0 * weights["MACD_exit"]
        macd_performance_score += 1 * weights["MACD_performance"]

    # Weak/Neutral bearish cross (MACD < signal, histogram increasing)
    elif macd < macd_signal and macd_hist_condition == "increasing":
        macd_entry_score += 0 * weights["MACD_entry"]
        macd_exit_score += 1 * weights["MACD_exit"]
        macd_performance_score += 1 * weights["MACD_performance"]

    # Neutral / no strong signal
    else:
        macd_entry_score += 0 * weights["MACD_entry"]
        macd_exit_score += 0 * weights["MACD_exit"]
        macd_performance_score += 0 * weights["MACD_performance"]

    log.info(
        f"MACD: {macd:.4f}, Signal: {macd_signal:.4f}, Hist: {macd_hist:.4f}, macd_trend_5: {macd_trend_5}, Entry Score: {macd_entry_score}, Exit Score: {macd_exit_score}, Performance Score: {macd_performance_score}"
    )

    # SMA30
    if above_sma:
        sma_entry_score += 1 * weights["SMA_entry"]
        sma_exit_score += 2 * weights["SMA_exit"]
        sma_performance_score += 1 * weights["SMA_performance"]
    else:
        sma_entry_score -= 1 * weights["SMA_entry"]
        sma_exit_score -= 2 * weights["SMA_exit"]
        sma_performance_score -= 1 * weights["SMA_performance"]
    log.info(
        f"SMA30: {above_sma}, Entry Score: {sma_entry_score}, Exit Score: {sma_exit_score}, Performance Score: {sma_performance_score}"
    )

    # Volume & PVT (simplified)
    if volume_condition == "high":
        vol_entry_score += 2 * weights["Volume_entry"]
        vol_exit_score += 1 * weights["Volume_exit"]
        vol_performance_score += 1 * weights["Volume_performance"]
    elif volume_condition == "low":
        vol_entry_score -= 1 * weights["Volume_entry"]
        vol_exit_score -= 1 * weights["Volume_exit"]
        vol_performance_score -= 1 * weights["Volume_performance"]
    else:
        vol_performance_score += 0  # normal volume, no score change
    log.info(
        f"Volume Condition: {volume_condition}, Entry Score: {vol_entry_score}, Exit Score: {vol_exit_score}, Performance Score: {vol_performance_score}"
    )

    if pvt_trend == "increasing":
        pvt_entry_score += 1 * weights["PVT_entry"]
        pvt_exit_score += 1 * weights["PVT_exit"]
        pvt_performance_score += 1 * weights["PVT_performance"]
    elif pvt_trend == "decreasing":
        pvt_entry_score -= 1 * weights["PVT_entry"]
        pvt_exit_score -= 1 * weights["PVT_exit"]
        pvt_performance_score -= 1 * weights["PVT_performance"]
    else:
        pvt_performance_score += 0  # stable PVT, no score change
    log.info(
        f"PVT Trend: {pvt_trend}, Entry Score: {pvt_entry_score}, Exit Score: {pvt_exit_score}, Performance Score: {pvt_performance_score}"
    )

    # Aggregate scores
    entry_score = (
        rsi_entry_score
        + macd_entry_score
        + sma_entry_score
        + vol_entry_score
        + pvt_entry_score
    )
    exit_score = (
        rsi_exit_score
        + macd_exit_score
        + sma_exit_score
        + vol_exit_score
        + pvt_exit_score
    )
    performance_score = (
        rsi_performance_score
        + macd_performance_score
        + sma_performance_score
        + vol_performance_score
        + pvt_performance_score
    )
    log.info(
        f"Total Entry Score: {entry_score}, Total Exit Score: {exit_score}, Total Performance Score: {performance_score}"
    )
    return performance_score, entry_score, exit_score


def classify(score, mode="entry"):
    """
    Classify a 0–1 score into human-readable labels.
    mode: "entry", "exit", or "performance"
    """

    if mode == "entry":
        if score >= 0.8:
            return "strong-buy"
        elif score >= 0.6:
            return "buy"
        elif score >= 0.4:
            return "neutral"
        elif score >= 0.2:
            return "cautious-buy"
        else:
            return "no-entry"

    elif mode == "exit":
        if score >= 0.8:
            return "strong-sell"
        elif score >= 0.6:
            return "sell"
        elif score >= 0.4:
            return "neutral"
        elif score >= 0.2:
            return "cautious-sell"
        else:
            return "no-exit"

    elif mode == "performance":
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "moderate"
        elif score >= 0.2:
            return "weak"
        else:
            return "poor"

    else:
        raise ValueError("mode must be 'entry', 'exit', or 'performance'")


# def classify(score, mode):
#     if mode == "performance":
#         return "good" if score >= 5 else "neutral" if score >= -1 else "poor"
#     elif mode == "entry":
#         return "strong buy" if score >= 5 else "buy" if score >= 2 else "neutral" if score >= -1 else "not now" if score >= -4 else "avoid"
#     else:
#         return "strong sell" if score >= 5 else "sell" if score >= 2 else "neutral" if score >= -1 else "not now" if score >= -4 else "avoid"


def sma30_scores(last):
    if last["Close"] > last["SMA30"]:
        return 1, 2, 1  # entry, exit, performance
    else:
        return -1, -2, -1


def compute_macd_scores(macd_line, signal_line, hist, trend="increasing"):
    """
    Translate raw MACD values + trend into weighted scores:
      • Entry Score (0–1): bullish entry attractiveness
      • Exit Score (0–1): bearish exit likelihood
      • Performance Score (0–1): momentum & signal quality
    """

    # --- Feature signals (normalized to -1 … +1) --- #
    crossover = 1 if macd_line > signal_line else -1
    hist_sign = 1 if hist > 0 else -1
    slope = 1 if trend == "increasing" else -1

    # Normalize divergence magnitude (0–1 scale)
    divergence = abs(macd_line - signal_line)
    magnitude = divergence / (abs(macd_line) + 1e-9)  # ratio
    magnitude = min(magnitude, 1.0)  # cap at 1

    # --- Entry Score weights --- #
    entry_weights = {
        "crossover": 0.4,
        "hist_sign": 0.3,
        "slope": 0.2,
        "magnitude": 0.1,
    }
    entry_score = (
        entry_weights["crossover"] * (crossover + 1) / 2
        + entry_weights["hist_sign"] * (hist_sign + 1) / 2
        + entry_weights["slope"] * (slope + 1) / 2
        + entry_weights["magnitude"] * magnitude
    )

    # --- Exit Score weights (inverse logic) --- #
    exit_weights = {
        "crossover": 0.4,
        "hist_sign": 0.3,
        "slope": 0.2,
        "magnitude": 0.1,
    }
    exit_score = (
        exit_weights["crossover"] * (1 - (crossover + 1) / 2)
        + exit_weights["hist_sign"] * (1 - (hist_sign + 1) / 2)
        + exit_weights["slope"] * (1 - (slope + 1) / 2)
        + exit_weights["magnitude"] * (1 - magnitude)
    )

    # --- Performance Score (quality of momentum) --- #
    perf_weights = {
        "magnitude": 0.5,
        "slope": 0.3,
        "hist_sign": 0.2,
    }
    performance_score = (
        perf_weights["magnitude"] * magnitude
        + perf_weights["slope"] * (slope + 1) / 2
        + perf_weights["hist_sign"] * (hist_sign + 1) / 2
    )

    return {
        "Entry Score": round(entry_score, 2),
        "Exit Score": round(exit_score, 2),
        "Performance Score": round(performance_score, 2),
    }


def compute_bollinger_scores(price, upper_band, lower_band, mid_band=None):
    """
    Translate Bollinger Band values into weighted scores:
      • Entry Score (0–1): attractiveness of long/short entry
      • Exit Score (0–1): likelihood of closing a position
      • Performance Score (0–1): strength of signal & volatility context
    """

    if mid_band is None:
        mid_band = (upper_band + lower_band) / 2

    # --- Normalize price position relative to bands (0 = lower, 1 = upper) --- #
    if upper_band == lower_band:
        pos = 0.5
    else:
        pos = (price - lower_band) / (upper_band - lower_band)
        pos = max(0, min(1, pos))  # clamp to [0,1]

    # --- Entry Score --- #
    # Strongest entry when price is near lower band (pos ~0)
    entry_score = 1 - pos  # closer to lower band = higher entry attractiveness

    # --- Exit Score --- #
    # Strongest exit when price is near upper band (pos ~1)
    exit_score = pos  # closer to upper band = higher exit likelihood

    # --- Performance Score --- #
    # Combination of volatility (bandwidth) and band extremes
    bandwidth = (upper_band - lower_band) / mid_band if mid_band != 0 else 0
    bandwidth = min(bandwidth, 1.0)  # normalize to [0,1]

    # Performance = how decisive the price move is
    if price > upper_band:  # breakout up
        performance_score = 0.8 + 0.2 * bandwidth
    elif price < lower_band:  # breakout down
        performance_score = 0.8 + 0.2 * bandwidth
    else:  # inside bands
        performance_score = 0.5 * (1 - abs(pos - 0.5)) + 0.5 * bandwidth

    return {
        "Entry Score": round(entry_score, 2),
        "Exit Score": round(exit_score, 2),
        "Performance Score": round(performance_score, 2),
    }


def compute_rsi_scores(rsi):
    """
    Translate RSI value into weighted scores:
      • Entry Score (0–1): attractiveness of long entry
      • Exit Score (0–1): likelihood of closing a position
      • Performance Score (0–1): strength of momentum signal
    """

    if rsi < 30:  # oversold
        entry_score = 1.0
        exit_score = 0.0
    elif rsi < 50:  # weak
        entry_score = min(1, (50 - rsi) / 20)  # fades as RSI approaches 30–50
        exit_score = 0.0
    elif rsi <= 70:  # neutral
        entry_score = max(0, (70 - rsi) / 20)  # fades as RSI approaches 50–70
        exit_score = max(0, (rsi - 50) / 20)  # fades as RSI approaches 50–70
    elif rsi < 80:  # strong
        entry_score = 0.0
        exit_score = min(1, (rsi - 70) / 10)  # fades as RSI approaches 70–80
    else:  # overbought
        entry_score = 0.0
        exit_score = 1.0

    performance_score = 1 - abs(rsi - 50) / 50  # best when RSI near extremes

    return {
        "Entry Score": round(entry_score, 2),
        "Exit Score": round(exit_score, 2),
        "Performance Score": round(performance_score, 2),
    }


def compute_sma_scores(price, sma):
    if price > sma:
        entry_score = 1
        exit_score = 0
        performance_score = 1
    else:
        entry_score = 0
        exit_score = 1
        performance_score = 0

    return {
        "Entry Score": round(entry_score, 2),
        "Exit Score": round(exit_score, 2),
        "Performance Score": round(performance_score, 2),
    }


def pvt_scores(pvt, upper_band, lower_band, mid_band=None):
    """
    Translate PVT (Price Volume Trend) + bands into standardized scores:
      • Entry Score (0–1): attractiveness of long entry
      • Exit Score (0–1): likelihood of bearish exit
      • Performance Score (0–1): strength/quality of the PVT trend
    """

    if mid_band is None:
        mid_band = (upper_band + lower_band) / 2

    # Normalize position of PVT within bands
    if upper_band == lower_band:
        pos = 0.5
    else:
        pos = (pvt - lower_band) / (upper_band - lower_band)
        pos = max(0, min(1, pos))  # clamp [0,1]

    # Entry Score → higher when PVT is above midline / upper band
    entry_score = pos

    # Exit Score → higher when PVT is near or below lower band
    exit_score = 1 - pos

    # Bandwidth → measure of volatility in PVT
    bandwidth = (upper_band - lower_band) / (mid_band if mid_band != 0 else 1e-9)
    bandwidth = min(bandwidth, 1.0)

    # Performance → strong if PVT breaks out of bands OR far from midline
    if pvt > upper_band or pvt < lower_band:
        performance_score = 0.8 + 0.2 * bandwidth  # breakout
    else:
        performance_score = 0.5 * (1 - abs(pos - 0.5)) + 0.5 * bandwidth

    return {
        "Entry Score": round(entry_score, 2),
        "Exit Score": round(exit_score, 2),
        "Performance Score": round(performance_score, 2),
    }


def check_signal(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 3:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    ticker = df.attrs["ticker"]

    # Crossovers
    crossed_up = (prev["MACD"] <= prev["MACD_SIGNAL"]) and (
        last["MACD"] > last["MACD_SIGNAL"]
    )

    # Price trend
    avg_price = df["Close"].rolling(SMA_WINDOW).mean()
    price_trend = "positive" if last["Close"] > avg_price.iloc[-1] else "negative"

    # Bollinger Bands
    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()
    upper_band = rolling_mean + rolling_std * 1.5
    lower_band = rolling_mean - rolling_std * 1.5
    last_close = last["Close"]
    if last_close > upper_band.iloc[-1]:
        band_position = "above band"
    elif last_close < lower_band.iloc[-1]:
        band_position = "below band"
    elif last_close > rolling_mean.iloc[-1]:
        band_position = "above median"
    elif last_close < rolling_mean.iloc[-1]:
        band_position = "below median"
    else:
        band_position = "at median"

    # RSI – Relative Strength Index

    rsi = 100 - (
        100
        / (
            1
            + (
                df["Close"].diff().fillna(0).clip(lower=0).rolling(window=14).mean()
                / df["Close"]
                .diff()
                .fillna(0)
                .clip(upper=0)
                .abs()
                .rolling(window=14)
                .mean()
            )
        )
    )
    last_rsi = rsi.iloc[-1] if not rsi.empty else None
    if last_rsi < 30:
        rsi_condition = "oversold"
    elif last_rsi < 50:
        rsi_condition = "weak"
    elif last_rsi <= 70:
        rsi_condition = "neutral"
    elif last_rsi > 70 and last_rsi < 80:
        rsi_condition = "strong"
    else:
        rsi_condition = "overbought"

    # PVT & Volume
    pvt = (df["Close"] - df["Close"].shift(1)).fillna(0) * df["Volume"]
    last_pvt = pvt.iloc[-1]
    pvt_trend = (
        "increasing" if last_pvt > 0 else "decreasing" if last_pvt < 0 else "stable"
    )
    avg_vol = df["Volume"].rolling(SMA_WINDOW).mean().iloc[-1]
    volume_condition = (
        "high"
        if last["Volume"] / avg_vol > 1.5
        else "low"
        if last["Volume"] / avg_vol < 0.5
        else "normal"
    )

    # MACD Histogram
    macd_hist_condition = (
        "increasing" if last["MACD_HIST"] > prev["MACD_HIST"] else "decreasing"
    )

    # MACD Trend over the last 5 periods
    macd_values = df["MACD"].tail(5).tolist()
    if len(macd_values) >= 5:
        if all(x < y for x, y in zip(macd_values, macd_values[1:])):
            macd_trend_5 = "increasing"
        elif all(x > y for x, y in zip(macd_values, macd_values[1:])):
            macd_trend_5 = "decreasing"
        else:
            macd_trend_5 = "stable"

    # Percentage above SMA30
    above_sma30_pct = (
        ((last["Close"] - last["SMA30"]) / last["SMA30"]) * 100
        if last["SMA30"] != 0
        else 0
    )
    above_sma50_pct = (
        ((last["Close"] - last["SMA50"]) / last["SMA50"]) * 100
        if last["SMA50"] != 0
        else 0
    )
    above_sma200_pct = (
        ((last["Close"] - last["SMA200"]) / last["SMA200"]) * 100
        if last["SMA200"] != 0
        else 0
    )

    # ----------  compute individual scores ----------
    log.info(f"---------- Computing Scores for {ticker} ---------")

    # compute MACD scores
    # scores = compute_macd_scores(60.78, 41.13, 19.65, trend="increasing")
    # print(scores)
    macd_scores_dict = compute_macd_scores(
        last["MACD"], last["MACD_SIGNAL"], last["MACD_HIST"], macd_hist_condition
    )
    log.info(f"MACD Scores: {macd_scores_dict}")

    # compute Bollinger scores
    bollinger_scores_dict = compute_bollinger_scores(
        last_close, upper_band.iloc[-1], lower_band.iloc[-1], rolling_mean.iloc[-1]
    )
    log.info(f"Bollinger Scores: {bollinger_scores_dict}")

    # compute RSI scores
    rsi_scores_dict = compute_rsi_scores(last_rsi)
    log.info(f"RSI Scores: {rsi_scores_dict}")

    # compute SMA30 scores
    sma30_scores_dict = compute_sma_scores(last["Close"], last["SMA30"])
    log.info(f"SMA30 Scores: {sma30_scores_dict}")

    # compute SMA50 scores
    sma50_scores_dict = compute_sma_scores(last["Close"], last["SMA50"])
    log.info(f"SMA50 Scores: {sma50_scores_dict}")

    # compute SMA200 scores
    sma200_scores_dict = compute_sma_scores(last["Close"], last["SMA200"])
    log.info(f"SMA200 Scores: {sma200_scores_dict}")

    # compute SMA scores
    sma_perfomance = (
        (
            sma30_scores_dict["Performance Score"]
            + sma50_scores_dict["Performance Score"]
            + sma200_scores_dict["Performance Score"]
        )
        / 3
        * 0.2
    )
    sma_entry = (
        sma30_scores_dict["Entry Score"]
        + sma50_scores_dict["Entry Score"]
        + sma200_scores_dict["Entry Score"]
    ) / 3
    sma_exit = (
        sma30_scores_dict["Exit Score"]
        + sma50_scores_dict["Exit Score"]
        + sma200_scores_dict["Exit Score"]
    ) / 3
    sma_scores_dict = {
        "Performance Score": round(sma_perfomance, 2),
        "Entry Score": round(sma_entry, 2),
        "Exit Score": round(sma_exit, 2),
    }
    log.info(f"SMA Combined Scores: {sma_scores_dict}")

    # compute PVT scores
    pvt_scores_dict = pvt_scores(
        last_pvt, upper_band.iloc[-1], lower_band.iloc[-1], rolling_mean.iloc[-1]
    )
    log.info(f"PVT Scores: {pvt_scores_dict}")

    # . EMA200 → 20%
    # • EMA30/50 crossover → 15%
    # • EMA30/50 price relation → 10%
    # • MACD Histogram → 15%
    # • MACD Signal Line → 10%
    # • RSI → 10%
    # • PVTBB → 10%
    # • Bollinger Bands → 10%
    ema200_weight = 0.2
    ema30_50_crossover_weight = 0.15
    ema30_50_price_weight = 0.1
    macd_hist_weight = 0.15
    macd_signal_weight = 0.1
    rsi_weight = 0.1
    pvt_weight = 0.1
    bollinger_weight = 0.1

    # aggregate scores with weights
    performance_score = (
        macd_scores_dict["Performance Score"]
        * weights["MACD_performance"]
        * macd_hist_weight
        + bollinger_scores_dict["Performance Score"]
        * weights["Bollinger_performance"]
        * bollinger_weight
        + rsi_scores_dict["Performance Score"] * weights["RSI_performance"] * rsi_weight
        + sma30_scores_dict["Performance Score"]
        * weights["SMA_performance"]
        * ema30_50_crossover_weight
        + sma50_scores_dict["Performance Score"]
        * weights["SMA_performance"]
        * ema30_50_price_weight
        + sma200_scores_dict["Performance Score"]
        * weights["SMA_performance"]
        * ema200_weight
        + pvt_scores_dict["Performance Score"] * weights["PVT_performance"] * pvt_weight
    )
    entry_score = (
        macd_scores_dict["Entry Score"] * weights["MACD_entry"] * macd_signal_weight
        + bollinger_scores_dict["Entry Score"]
        * weights["Bollinger_entry"]
        * bollinger_weight
        + rsi_scores_dict["Entry Score"] * weights["RSI_entry"] * rsi_weight
        + sma30_scores_dict["Entry Score"]
        * weights["SMA_entry"]
        * ema30_50_crossover_weight
        + sma50_scores_dict["Entry Score"]
        * weights["SMA_entry"]
        * ema30_50_price_weight
        + sma200_scores_dict["Entry Score"] * weights["SMA_entry"] * ema200_weight
        + pvt_scores_dict["Entry Score"] * weights["PVT_entry"] * pvt_weight
    )
    exit_score = (
        macd_scores_dict["Exit Score"] * weights["MACD_exit"] * macd_signal_weight
        + bollinger_scores_dict["Exit Score"]
        * weights["Bollinger_exit"]
        * bollinger_weight
        + rsi_scores_dict["Exit Score"] * weights["RSI_exit"] * rsi_weight
        + sma30_scores_dict["Exit Score"]
        * weights["SMA_exit"]
        * ema30_50_crossover_weight
        + sma50_scores_dict["Exit Score"] * weights["SMA_exit"] * ema30_50_price_weight
        + sma200_scores_dict["Exit Score"] * weights["SMA_exit"] * ema200_weight
        + pvt_scores_dict["Exit Score"] * weights["PVT_exit"] * pvt_weight
    )
    # summarize performance scores
    log.info(f"Performance Scores: {performance_score}")
    log.info(f"Entry Scores: {entry_score}")
    log.info(f"Exit Scores: {exit_score}")

    # Scores
    # performance_score, entry_score, exit_score = compute_scores(ticker, last, macd_hist_condition, price_trend,
    #                                                            band_position, above_sma, volume_condition,
    #                                                            pvt_trend, macd_trend_5, weights)
    performance_state = classify(performance_score, mode="performance")
    entry_state = classify(entry_score, mode="entry")
    exit_state = classify(exit_score, mode="exit")

    # Yahoo recommendation
    yahoo_rec = get_recommendation(df.attrs["ticker"])

    ticker_name = get_fund_name(df.attrs["ticker"])

    payload = {
        "ticker": ticker,
        "title": ticker_name,
        "ycp": f"{last['YCP']:.2f}",
        "price": f"{last['Close']:.2f}",
        "performance": performance_state,
        "entry": entry_state,
        "exit": exit_state,
        "price_trend": price_trend,
        "bollinger": band_position,
        "rsi_condition": rsi_condition,
        "volume": volume_condition,
        "pvt_trend": pvt_trend,
        "macd_trend_5": macd_trend_5,
        "signal": f"{last['MACD_SIGNAL']:.4f}",
        "histogram": f"{last['MACD_HIST']:.4f}",
        "crossover": "Yes" if crossed_up else "No",
        "sma30": f"{above_sma30_pct:.2f}%",
        "sma50": f"{above_sma50_pct:.2f}%",
        "sma200": f"{above_sma200_pct:.2f}%",
        "yahoo_verdict": yahoo_rec.get("verdict", "N/A"),
    }

    return payload


def get_fund_name(ticker: str) -> str:
    """
    Fetch the full fund/company name from Yahoo Finance given a ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return (
            info.get("longName")
            or info.get("shortName")
            or f"No name found for {ticker}"
        )
    except Exception as e:
        return f"Error fetching data for {ticker}: {e}"


# ---------- Print Notification ----------
def print_notification(payload):
    # print nicely formatted notification with : "title", "performance", "entry", "exit"
    print("=" * 120)
    print(
        f"Notification for {payload['title']} - Performance: {payload['performance']} , Entry: {payload['entry']} , Exit: {payload['exit']}"
    )
    print("=" * 120)
    for key, value in payload.items():
        if key not in ["title", "performance", "entry", "exit"]:
            print(f"  {key}: {value}")


def render_html(payload) -> None:
    """
    Generate an html table from the payload dict and write to file index.html

    """
    # Keys as columns, values as rows in the HTML table
    # loop through payload and append HTML to file index.html
    html = render_html_table(payload, title="Stock Signals", output_file='./docs/index.html')
    print("Wrote index.html with", len(payload), "rows")

    with open("index.html", "w") as f:
        f.write(html)


# ---------- Push Notification ----------
async def send_push(payloads):
    if not PUSH or not APP_TOKEN or not USER_KEY:
        print("Push disabled or missing credentials.")
        return
    url = "https://api.pushover.net/1/messages.json"
    async with aiohttp.ClientSession() as session:
        for payload in payloads:
            try:
                async with session.post(
                    url,
                    data={
                        "token": APP_TOKEN,
                        "user": USER_KEY,
                        "title": payload["title"],
                        "message": f"Price = {payload['price']} \n Performance = {payload['performance'].upper()} \n Entry = {payload['entry'].upper()} \n Exit = {payload['exit'].upper()} \n Yahoo Verdict = {payload['yahoo_verdict'].upper()} \n",
                        "html": 1,
                    },
                ) as resp:
                    print(f"Pushover sent for {payload['title']}: {resp.status}")
            except Exception as e:
                print(f"Push error for {payload['title']}: {e}")


# ---------- Main Loop ----------
async def analyze_and_alert(tickers):
    payloads = []
    loop = asyncio.get_running_loop()
    for ticker in tickers:
        try:
            # run blocking yfinance fetch in executor
            df = await loop.run_in_executor(None, fetch_history, ticker)
            log.info(f"Fetched data for {ticker}, {len(df)} rows")
            if df.empty:
                log.warning(f"No data for {ticker}, skipping.")
                continue
            df_ind = compute_indicators(df)
            log.info(f"Computed indicators for {ticker}")
            payload = check_signal(df_ind)
            log.info(f"Checked signals for {ticker}")
            if payload:
                payloads.append(payload)
                log.info(f"Generated payload for {ticker}")

        except Exception as e:
            log.error(f"Error processing {ticker}: {e}")

    if PUSH and payloads:
        await send_push(payloads)
    elif payloads:
        render_html(payloads)


if __name__ == "__main__":
    configure_logging()
    log.info("Starting stock analysis...")

    # open payload.json and render html table
    # if os.path.exists("payload.json"):
    #     import json
    #     with open("payload.json", "r") as f:
    #         payloads = json.load(f)
    #     render_html(payloads)
    asyncio.run(analyze_and_alert(TICKERS))
