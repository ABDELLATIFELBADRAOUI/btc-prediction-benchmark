# raw/

Vendor exports live here. **Nothing in this directory is published** — see the
`.gitignore` rules. The providers' terms do not permit redistribution.

```
raw/
├── cmc_exports/               the CoinMarketCap CSV segments (any number)
├── btc_ohlcv.csv              built by scripts/prepare_raw.py
└── google_trends_monthly.csv  84 rows, date + google_trends
```

Pipeline:

```bash
python scripts/prepare_raw.py    --indir raw/cmc_exports --out raw/btc_ohlcv.csv
python scripts/build_features.py --ohlcv raw/btc_ohlcv.csv \
                                 --trends raw/google_trends_monthly.csv
python scripts/verify_dataset.py
```

The CoinMarketCap export is capped at 400 rows per request, so the study window
must be pulled in overlapping segments. Overlap is harmless — `prepare_raw.py`
deduplicates by date and discards anything outside 2017-01-01..2023-12-31.
Segment coverage should leave exactly two calendar gaps, 2019-11-26 and
2021-11-25, which are filled by linear interpolation; the script warns if your
gap set differs.
