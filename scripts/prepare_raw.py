#!/usr/bin/env python3
"""
prepare_raw.py -- assemble the segmented CoinMarketCap exports into a single
daily OHLCV series that feeds build_features.py. By default it builds the
2,556-row series of the manuscript (2017-01-01 to 2023-12-31); --start and
--end select another window.

Why this step exists
--------------------
The CoinMarketCap web export is capped at 400 rows per request, so the
2017-01-01 to 2023-12-31 window has to be pulled in overlapping segments and
stitched. The segment boundaries leave exactly two one-day calendar gaps,
2019-11-26 and 2021-11-25, which are filled by linear interpolation between
adjacent observations. This reproduces the procedure described in the Data
section of the manuscript.

Input
-----
A directory of CoinMarketCap CSV exports. Expected format (their default):
semicolon-separated, UTF-8 with BOM, columns
    timeOpen;timeClose;timeHigh;timeLow;name;open;high;low;close;volume;marketCap;circulatingSupply;timestamp
Overlapping segments are fine -- the script checks that they agree on every
column and then drops the duplicates. Rows outside the
requested window are discarded, so extra exports do no harm.

Output
------
raw/btc_ohlcv.csv -- 2,556 rows, columns
    date,open,high,low,close,volume,market_cap
For another window the default output is raw/btc_ohlcv_<start>_<end>.csv, so
the series of the manuscript is never overwritten.

Usage
-----
    python scripts/prepare_raw.py --indir raw/cmc_exports --out raw/btc_ohlcv.csv
    python scripts/prepare_raw.py --indir raw/cmc_exports --end 2026-11-12
    python scripts/build_features.py --ohlcv raw/btc_ohlcv.csv \
                                     --trends raw/google_trends_monthly.csv
    python scripts/verify_dataset.py

Note on precision
-----------------
OHLC prices are rounded to 2 decimals, matching the published dataset. The
export carries full float32 precision (e.g. 920.958984375); the dataset behind
the manuscript stores 920.96. Without this rounding the features differ in the
sixth significant figure and verify_dataset.py will fail.
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

START = "2017-01-01"
END = "2023-12-31"
EXPECTED_ROWS = 2556
KNOWN_GAPS = ["2019-11-26", "2021-11-25"]


def load_exports(indir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(indir / "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {indir}")
    frames = []
    for f in files:
        d = pd.read_csv(f, sep=";", encoding="utf-8-sig")
        if "timeOpen" not in d.columns:
            print(f"  skipped (unexpected format): {Path(f).name}")
            continue
        d["date"] = pd.to_datetime(d["timeOpen"]).dt.tz_localize(None).dt.normalize()
        frames.append(d[["date", "open", "high", "low", "close", "volume", "marketCap"]])
        print(f"  {Path(f).name:64s} {len(d):4d} rows  "
              f"{d['date'].min().date()} -> {d['date'].max().date()}")
    if not frames:
        raise SystemExit("No usable exports.")
    return pd.concat(frames)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", required=True, type=Path,
                    help="directory containing the CoinMarketCap CSV exports")
    ap.add_argument("--start", default=START,
                    help=f"first day of the series (default {START})")
    ap.add_argument("--end", default=END,
                    help=f"last day of the series (default {END})")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default raw/btc_ohlcv.csv for the default "
                         "window, raw/btc_ohlcv_<start>_<end>.csv otherwise)")
    a = ap.parse_args()
    start, end = str(pd.Timestamp(a.start).date()), str(pd.Timestamp(a.end).date())
    default_window = (start, end) == (START, END)
    if a.out is None:
        name = "btc_ohlcv.csv" if default_window else f"btc_ohlcv_{start}_{end}.csv"
        a.out = Path(__file__).resolve().parent.parent / "raw" / name

    print(f"Reading exports from {a.indir}")
    raw = load_exports(a.indir)

    window = raw[(raw["date"] >= start) & (raw["date"] <= end)]
    cols = ["open", "high", "low", "close", "volume", "marketCap"]
    counts = window.groupby("date").size()
    spread = window.groupby("date")[cols].agg(lambda x: x.max() - x.min())
    n_disagree = int((spread > 0).any(axis=1).sum())
    print(f"\ndays covered by more than one export: {int((counts > 1).sum())}; "
          f"days on which the exports disagree: {n_disagree}")
    if n_disagree:
        print("  WARNING: overlapping exports differ; the first file in name order is kept.")

    raw = (raw.drop_duplicates(subset="date", keep="first")
              .rename(columns={"marketCap": "market_cap"})
              .set_index("date").sort_index()
              .loc[start:end])
    print(f"\nconcatenated and deduplicated: {len(raw)} distinct days")

    full = raw.reindex(pd.date_range(start, end, freq="D"))
    gaps = [str(d.date()) for d in full.index[full["close"].isna()]]
    known = [g for g in KNOWN_GAPS if start <= g <= end]
    print(f"calendar gaps found: {gaps or 'none'}")
    if sorted(gaps) != sorted(known):
        print(f"  WARNING: expected exactly {known or 'no gap'}.")
        print("  A different gap set means your export coverage differs from ours,")
        print("  and the published dataset will not reproduce.")
    full = full.interpolate(method="linear")

    for c in ("open", "high", "low", "close"):
        full[c] = full[c].round(2)          # match the published precision

    full.index.name = "date"
    print(f"final series: {len(full)} rows, {full.index.min().date()} -> {full.index.max().date()}")
    if default_window and len(full) != EXPECTED_ROWS:
        print(f"  WARNING: expected {EXPECTED_ROWS} rows.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(a.out, index_label="date")
    print(f"written: {a.out}")
    print("\nNext:  python scripts/build_features.py --ohlcv "
          f"{a.out} [--trends <monthly_trends.csv>]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
