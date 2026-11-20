# Directional accuracy under three definitions: code and results

Code and result tables for the manuscript **"Directional Accuracy Is Three Different Measures: Closed Forms, No-Skill Reference Values and a Bitcoin Application"** by Abdellatif Elbadraoui, Yassine Mouhssine, Abdelkader El Alaoui and Said Ouatik El Alaoui (submitted).

The notebook compares nine one-day-ahead forecasts of the Bitcoin daily close over 1 January 2024 to 11 November 2026 (1,046 test days): the random walk and eight learned models. It computes directional accuracy in three ways (on the traded signal, on the change between consecutive price forecasts and on the change between consecutive return forecasts), scores forecasts that use no model in the same way, and runs the Monte Carlo experiments, the positive control and the calibrated tests of the manuscript.

Release **v1.1.0** contains everything behind the numbers, tables and figures of the manuscript. Release v1.0.0 ([10.5281/zenodo.21827665](https://doi.org/10.5281/zenodo.21827665)) archived an earlier version of the study (notebook v16, data from 2017 to 2023 with Google Trends); its files remain available under the tag [v1.0.0](https://github.com/ABDELLATIFELBADRAOUI/btc-prediction-benchmark/tree/v1.0.0).

## Layout

```
notebooks/bitcoin_prediction_benchmark_v19.ipynb   analysis notebook, with the outputs of the reported run
results/v19/                                       result tables (t01 to t16) and manifest_v19.json
figures/v19/                                       figures drawn by the notebook
figures/paper/                                     Figures 1 to 3 of the manuscript
scripts/prepare_raw.py                             assembles the CoinMarketCap exports into one daily OHLCV series
scripts/build_features.py                          computes the 19 features and the next-day close
scripts/verify_dataset.py                          checks the 2017-2023 dataset of release v1.0.0
scripts/make_v19.py                                writes the notebook from its cell definitions
scripts/verify_v19.py                              recomputes headline numbers from the saved forecasts with its own code
scripts/sign_test_power.py                         power of the exact sign test and its p-values under other references
scripts/make_figures.py                            draws Figures 1 to 3 of the manuscript
data/, raw/                                        inputs, not redistributed (see data/README.md and raw/README.md)
```

## Data

The price data are not redistributed: the provider's terms of use do not permit it. To rebuild the dataset:

1. Export the daily Bitcoin history from CoinMarketCap (https://coinmarketcap.com) for 1 January 2017 to 12 November 2026, in segments, into `raw/cmc_exports/` (details in `raw/README.md`).
2. Download the monthly 3-month Treasury bill rate (FRED series TB3MS, https://fred.stlouisfed.org/series/TB3MS) as `raw/fred_tb3ms.csv`, with the columns `observation_date,TB3MS`.
3. Build the daily series and the features:

   ```bash
   python scripts/prepare_raw.py --indir raw/cmc_exports --end 2026-11-12
   python scripts/build_features.py --ohlcv raw/btc_ohlcv_2017-01-01_2026-11-12.csv
   ```

   The first command writes `raw/btc_ohlcv_2017-01-01_2026-11-12.csv` (3,603 days; the two days missing from the export, 2019-11-26 and 2021-11-25, are interpolated). The second writes `data/btc_features_2017-01-01_2026-11-12_no_trends.csv` (3,572 rows, 2017-01-31 to 2026-11-11, 19 features, no Google Trends). Without `--end`, the two scripts build the 2017-2023 dataset of release v1.0.0.
4. The notebook prints a fingerprint of the dataset (number of rows, first and last dates, moments of five columns) and stores it in `results/v19/manifest_v19.json` under `data_fingerprint`. A rebuilt dataset should give the same values.

## Running the analysis

```bash
pip install -r requirements.txt jupyter      # Python 3.11
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
        notebooks/bitcoin_prediction_benchmark_v19.ipynb
```

Paths resolve from the notebook's location; the environment variable `BTC_BENCHMARK_ROOT` overrides them. The reported run took about 38 minutes on two processor cores (`results/v19/manifest_v19.json` records the run, the settings and the library versions). `BTC_FAST=1` runs a reduced version to check the pipeline; its numbers are not those of the manuscript.

The notebook writes the result tables to `results/v19/` and its figures to `figures/v19/`. Forecasts in price units go to `results/v19/_local/`, which is not published because it contains prices. Each table is described in the notebook cell that writes it.

Then, from the repository root:

```bash
python scripts/verify_v19.py                    # needs data/ and results/v19/_local/
python scripts/sign_test_power.py results/v19   # needs results/v19/ only
python scripts/make_figures.py                  # Figures 1 to 3; Fig. 2 needs data/ and results/v19/_local/
```

`verify_v19.py` recomputes the headline numbers from the saved forecasts with its own code and reports how many differ from the tables (none for the reported run). `sign_test_power.py` computes, from the counts in `results/v19/t05_direction.csv`, the power of the exact sign test and its p-values when a rise has a probability of 0.51 or 0.49.

The deep-learning models use the seeds 42, 123 and 456. Two complete runs on the same machine gave identical model tables; on other hardware or library builds, the results of the deep-learning models can differ slightly.

## Citation

Please cite the archived software through its concept DOI, which always resolves to the latest version: [10.5281/zenodo.21827664](https://doi.org/10.5281/zenodo.21827664). `CITATION.cff` gives the metadata.

## License

MIT for the code (see `LICENSE`). The license does not cover the price data, which are not included.
