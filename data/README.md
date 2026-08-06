# data/

**This directory is intentionally empty in the public release.**

The Bitcoin price series and the Google Trends attention series are not
redistributed here: the providers' terms of use do not permit republication of
their exports. What is released instead is everything needed to rebuild the
dataset and to prove that your rebuild is identical to ours.

## To reproduce the benchmark

1. **Acquire the raw price data.** Export daily BTC OHLCV (open, high, low,
   close, volume, market cap) for 2017-01-01 to 2023-12-31 from CoinMarketCap
   (https://coinmarketcap.com). The export is capped per request, so it must be
   taken in segments and concatenated chronologically; the concatenated file has
   2,556 daily rows. Two dates absent from the platform export (2019-11-26 and
   2021-11-25) are filled by linear interpolation between adjacent observations.

2. **Acquire the attention series.** Download Google Trends interest for the
   query "bitcoin" over the same period at native monthly frequency
   (84 months) from https://trends.google.com, then linearly interpolate to
   daily. The manuscript is explicit that this is a low-pass monthly proxy, not
   a true daily signal; the native daily variant used in robustness check B4 is
   fetched at runtime by the notebook via `pytrends`.

3. **Build the feature file.** Run `scripts/build_features.py` (see below) to
   produce `data/btc_features_final.csv`: 2,525 rows after the rolling-window
   computations, 20 features, no missing values.

4. **Verify before you run anything.**

   ```bash
   python scripts/verify_dataset.py data/btc_features_final.csv
   ```

   This checks row count, date range, schema and the moments of every input
   series against the values published in Table 2 of the manuscript. If it
   passes, your dataset is the one the paper was computed from.

## On `build_features.py`

`scripts/build_features.py` rebuilds the feature file:

```bash
python scripts/build_features.py --ohlcv raw/btc_ohlcv.csv \
                                 --trends raw/google_trends_monthly.csv
python scripts/verify_dataset.py
```

Every formula in it was verified by recomputing each column from the released
file's own OHLC and comparing: all fifteen derived columns reproduce to
floating-point precision (max relative error 1.1e-12).

Two definitions are easy to get wrong and are flagged in the source:

- **`rsi_14` uses a simple 14-day rolling mean** of gains and losses, not
  Wilder's `ewm(alpha=1/14)` smoothing. The Wilder variant differs by several
  RSI points and will not reproduce the published numbers.
- **`bb_width` is normalised by the 20-day middle band**, not by the close.

Bollinger bands use a 20-day window at 2 standard deviations with pandas'
default `ddof=1`. The head trim comes from `dropna()` after the rolling windows
and the tail row from the undefined next-day target: 2556 - 30 - 1 = 2525 rows,
2017-01-31 to 2023-12-30.

## On the google_trends column

Repeated Google Trends downloads of the same window return slightly different
sampled values. `verify_dataset.py` therefore allows a 2% tolerance on that
column and none on the others. A failure confined to `google_trends` means your
attention series differs from ours within the provider's own sampling
variability; the price-derived results are unaffected.

## Sources

- OHLCV — CoinMarketCap, https://coinmarketcap.com
- Google Trends — https://trends.google.com and the `pytrends` API
- On-chain metrics (robustness check B1) — CoinMetrics community data,
  https://docs.coinmetrics.io — fetched at runtime by the notebook
