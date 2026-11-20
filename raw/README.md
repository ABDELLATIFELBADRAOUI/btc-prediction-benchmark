# raw/

The vendor exports live here. **Nothing in this directory except this file is
published** (see `.gitignore`): the provider's terms do not permit redistribution.

```
raw/
├── cmc_exports/                           the CoinMarketCap CSV segments (any number)
├── btc_ohlcv_2017-01-01_2026-11-12.csv    built by scripts/prepare_raw.py
└── fred_tb3ms.csv                         FRED series TB3MS: observation_date,TB3MS
```

Pipeline for release v1.1.0:

```bash
python scripts/prepare_raw.py    --indir raw/cmc_exports --end 2026-11-12
python scripts/build_features.py --ohlcv raw/btc_ohlcv_2017-01-01_2026-11-12.csv
```

The CoinMarketCap export is capped at 400 rows per request, so the window from
2017-01-01 to 2026-11-12 must be pulled in overlapping segments. Overlap is harmless:
`prepare_raw.py` checks that overlapping segments agree, removes duplicate dates and
discards anything outside the window. The segments should leave exactly two calendar
gaps, 2019-11-26 and 2021-11-25, which are filled by linear interpolation; the script
warns if the gaps differ. The output has 3,603 daily rows.

`fred_tb3ms.csv` is the monthly 3-month Treasury bill rate downloaded from
https://fred.stlouisfed.org/series/TB3MS, with the columns `observation_date,TB3MS`.
