# data/

**This directory is empty in the public repository.** The Bitcoin price series is not
redistributed: the provider's terms of use do not permit republication of its exports.
What is released instead is the code that rebuilds the dataset from them.

## Dataset of release v1.1.0

`data/btc_features_2017-01-01_2026-11-12_no_trends.csv`: 3,572 daily rows,
2017-01-31 to 2026-11-11, with the open, high, low and close prices, the volume, the
market capitalisation, 13 features computed from them (19 features in all, no Google
Trends), the next-day close and its direction. Build it from the CoinMarketCap
exports (see
`raw/README.md`):

```bash
python scripts/prepare_raw.py --indir raw/cmc_exports --end 2026-11-12
python scripts/build_features.py --ohlcv raw/btc_ohlcv_2017-01-01_2026-11-12.csv
```

The notebook stores a fingerprint of the dataset (number of rows, first and last dates,
moments of five columns) in `results/v19/manifest_v19.json` under `data_fingerprint`;
a rebuilt dataset should give the same values.

Two definitions are easy to get wrong and are flagged in `scripts/build_features.py`:
`rsi_14` uses a simple 14-day rolling mean of gains and losses, not Wilder's smoothing,
and `bb_width` is normalised by the 20-day middle band, not by the close. The first 30
days are used by the rolling windows, and the last day has no next-day close.

## Dataset of release v1.0.0

Without `--end` and with `--trends`, the two scripts build `data/btc_features_final.csv`
(2017 to 2023, with Google Trends), the input of notebook v16 in release v1.0.0;
`scripts/verify_dataset.py` checks it.

## Sources

- Daily open, high, low, close, volume and market capitalisation: CoinMarketCap,
  https://coinmarketcap.com
- 3-month Treasury bill rate (risk-free rate of the trading results): FRED series TB3MS,
  https://fred.stlouisfed.org/series/TB3MS
