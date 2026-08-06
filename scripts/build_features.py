#!/usr/bin/env python3
"""
build_features.py -- rebuild btc_features_final.csv from the primary sources.

The vendor exports are not redistributed with this repository (their terms of use
do not permit it), so this script plus scripts/verify_dataset.py is what lets a
reader reconstruct the exact dataset behind every number in the manuscript.

Inputs
------
  --ohlcv    daily BTC OHLCV export, 2017-01-01 to 2023-12-31, 2556 rows.
             Columns: date, open, high, low, close, volume, market_cap
             Source: CoinMarketCap. The export is capped per request, so it is
             taken in segments and concatenated chronologically. Two dates
             absent from the platform export (2019-11-26, 2021-11-25) are filled
             by linear interpolation between adjacent observations.

  --trends   Google Trends interest for the query "bitcoin", native MONTHLY
             frequency (84 rows), 2017-01 to 2023-12, linearly interpolated to
             daily by this script.
             Columns: date, google_trends

Output
------
  data/btc_features_final.csv -- 2525 rows, 2017-01-31 to 2023-12-30,
  20 features + target_price + target_direction, no missing values.

Usage
-----
    python scripts/build_features.py --ohlcv raw/btc_ohlcv.csv \\
                                     --trends raw/google_trends_monthly.csv
    python scripts/verify_dataset.py            # confirm it matches the paper

FORMULA PROVENANCE
------------------
Every formula below was verified against the released btc_features_final.csv by
recomputing it from the file's own OHLC columns. All columns reproduce to
floating-point precision after the EMA warm-up. Two definitions are easy to get
wrong and are flagged inline:

  * rsi_14   uses a SIMPLE 14-day rolling mean of gains and losses, not Wilder's
             exponential smoothing. The Wilder variant differs by several RSI
             points and will not reproduce the published numbers.
  * bb_width is normalised by the 20-day MIDDLE band, not by the close.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RSI_PERIOD = 14
BB_WINDOW = 20
BB_K = 2


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI with a simple rolling mean of gains/losses.

    NOTE: this is the arithmetic-mean variant, NOT Wilder's ewm(alpha=1/period)
    smoothing. Verified against the published file; the Wilder variant does not
    reproduce it.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def build_features(ohlcv: pd.DataFrame, trends_daily: pd.Series) -> pd.DataFrame:
    """Compute the 20 predictors and the target from raw OHLCV + daily trends."""
    df = ohlcv.copy()
    df["google_trends"] = trends_daily.reindex(df.index)

    close = df["close"]

    # --- volatility block (3) -------------------------------------------------
    df["log_return"] = np.log(close / close.shift(1))
    df["volatility_7d"] = df["log_return"].rolling(7).std()
    df["volatility_30d"] = df["log_return"].rolling(30).std()

    # --- technical block (8) --------------------------------------------------
    df["rsi_14"] = rsi(close)
    df["ema_7"] = close.ewm(span=7, adjust=False).mean()
    df["ema_30"] = close.ewm(span=30, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    roll = close.rolling(BB_WINDOW)
    bb_mid, bb_sd = roll.mean(), roll.std()          # pandas default ddof=1
    df["bb_upper"] = bb_mid + BB_K * bb_sd
    df["bb_lower"] = bb_mid - BB_K * bb_sd
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid   # /mid, not /close

    # --- price-level ratios (2 of the 8) --------------------------------------
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_open_ratio"] = df["close"] / df["open"]

    # --- target ---------------------------------------------------------------
    df["target_price"] = close.shift(-1)
    df["target_direction"] = (df["target_price"] > close).astype(int)

    # Drop the rolling-window warm-up at the head and the undefined target at the
    # tail: 2556 - 30 - 1 = 2525 rows, 2017-01-31 to 2023-12-30.
    df = df.dropna()

    ordered = ["open", "high", "low", "close", "volume", "market_cap",
               "google_trends", "log_return", "volatility_7d", "volatility_30d",
               "rsi_14", "ema_7", "ema_30", "macd", "macd_signal",
               "bb_upper", "bb_lower", "bb_width",
               "high_low_ratio", "close_open_ratio",
               "target_price", "target_direction"]
    return df[ordered]


def load_trends_monthly_to_daily(path: Path, index: pd.DatetimeIndex) -> pd.Series:
    """Read the native monthly Trends export and linearly interpolate to daily.

    The manuscript is explicit that this yields a low-pass MONTHLY attention
    proxy, not a true daily signal.
    """
    t = pd.read_csv(path)
    date_col = t.columns[0]
    val_col = [c for c in t.columns if c != date_col][0]
    t[date_col] = pd.to_datetime(t[date_col])
    s = t.set_index(date_col)[val_col].astype(float).sort_index()
    return s.reindex(s.index.union(index)).interpolate(method="linear").reindex(index)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ohlcv", required=True, type=Path)
    ap.add_argument("--trends", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "btc_features_final.csv")
    a = ap.parse_args()

    ohlcv = pd.read_csv(a.ohlcv, parse_dates=["date"]).set_index("date").sort_index()
    missing = {"open", "high", "low", "close", "volume", "market_cap"} - set(ohlcv.columns)
    if missing:
        print(f"OHLCV file is missing columns: {sorted(missing)}")
        return 1
    print(f"raw OHLCV : {len(ohlcv)} rows, {ohlcv.index.min().date()} -> {ohlcv.index.max().date()}")
    if len(ohlcv) != 2556:
        print(f"  WARNING: expected 2556 rows; the published dataset will not reproduce")

    trends = load_trends_monthly_to_daily(a.trends, ohlcv.index)
    print(f"trends    : interpolated to {trends.notna().sum()} daily observations")

    feats = build_features(ohlcv, trends)
    print(f"features  : {len(feats)} rows x {feats.shape[1]} columns, "
          f"{feats.index.min().date()} -> {feats.index.max().date()}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(a.out, index=True, index_label="date")
    print(f"written   : {a.out}")
    print("\nNow run:  python scripts/verify_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
