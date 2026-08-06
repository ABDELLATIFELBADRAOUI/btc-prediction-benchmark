# Bitcoin price prediction — a reproducible benchmark

Code and data for **"Evaluating Machine Learning Models for Bitcoin Price Prediction: A Reproducible Benchmark with Formal Statistical Testing and Transaction-Cost Sensitivity"** (submitted to *SN Computer Science*).

Seven models across four families are compared on seven years of daily BTC data (2017–2023) under a strict temporal split, with HLN-corrected Diebold–Mariano tests, Pesaran–Timmermann directional-skill tests, multi-seed reporting for the deep models, and a transaction-cost-aware trading simulation.

The headline finding — the **Directional Accuracy Pitfall** — is that on this data, error metrics, directional accuracy and trading utility measure different things: no model shows PT-detectable directional skill, and the best-RMSE model has nearly the worst Sharpe ratio.

\---

## Layout

```
btc-prediction-benchmark/
├── data/          btc\_features\_final.csv          (input — see data/README.md)
├── notebooks/     bitcoin\_prediction\_benchmark\_v16.ipynb
├── figures/       \*.png                           (generated)
├── results/       \*.csv, manifest.json            (generated)
├── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
jupyter lab notebooks/bitcoin\_prediction\_benchmark\_v16.ipynb
```

Run top to bottom. Paths resolve from the notebook's location; override with `BTC\_BENCHMARK\_ROOT` if you keep the data elsewhere. Phases 9 and 10 (on-chain and daily Google Trends robustness checks) need internet access and degrade gracefully without it.

Total runtime is a few minutes on a laptop CPU — the deep models are small (≈25k–36k parameters) and each takes 98–200 s per seed.

**Phase 12 must run clean.** It cross-checks the figures against the tables, verifies that per-regime RMSEs pool back to the full-sample RMSE, and confirms every manuscript figure exists. If it reports failures, do not archive the artefact or compile the paper.

## What each phase produces

|Phase|Output|Used in|
|-|-|-|
|1|descriptive statistics, stationarity|Tables 2, 5; Fig. 1|
|2|Random Walk, Random Forest, XGBoost + grid search|Table 6|
|3|CNN-LSTM, GRU+Att, two walk-forward variants (3 seeds)|Tables 6, 8|
|4|HLN-corrected DM tests, Pesaran–Timmermann tests|Tables 6, 7; Fig. 3|
|5|long/short simulation at 0/10/50 bp, DA-vs-Sharpe scatter|Table 9; Figs. 4, 5, 6|
|6|split-conformal intervals|**not reported — see below**|
|7|TreeSHAP, feature ablation|Tables 10, 11; Figs. 7, 8|
|8|multi-horizon, per-regime analysis|Tables 12, 13; Figs. 9, 10|
|9–10|on-chain (B1) and daily Google Trends (B4) enrichments|§5.9|
|11–12|manifest, summary, verification|—|

## Known limitations of this artefact

**Phase 6 (split-conformal) is not reported in the paper.** Its calibration set is in-sample: `xgb\_final` is refitted on train ∪ validation, so scoring `X\_val` re-scores data the model has seen. The residuals are memorised, the calibration quantile collapses, and empirical coverage falls to about 3% against a 90% nominal level. Doing this correctly requires fitting on the training window only and holding 2021 out as a genuine calibration set — a different experiment. The cells are retained for transparency but no conformal result should be quoted from them.

**B4 (daily Google Trends) is not stable across fetches.** Google Trends normalises each requested window separately and returns slightly different sampled values on repeat calls, so the stitched daily series carries chunk-boundary scale artefacts and re-running Phase 10 will not reproduce the stored numbers exactly. The RMSE effect of this enrichment is small and not stable in sign; §5.9 of the paper reports it as such. Two outcomes are stable: DA does not improve, and `google\_trends` drops in the SHAP ranking.

**The deep models are run in standard configurations, not tuned.** XGBoost is selected by exhaustive grid search over 27 configurations; the CNN-LSTM and GRU+Att are not searched. Part of the RMSE gap may therefore reflect search budget rather than architecture. This is stated as a limitation in the paper rather than papered over. Note also that the XGBoost grid search is only weakly identifying: all 27 configurations fall within about 3% of each other on a validation metric an order of magnitude worse than the eventual test RMSE, because 2021 sits far above the training price level.

**Early stopping monitors the training loss.** The deep models are fitted directly on the combined train ∪ validation window with `monitor='loss'`; there is no separate validation-loss monitoring for these architectures.

## Changes from v15

1. Portable paths — no absolute Windows path anywhere.
2. `fig3\_dm\_forest.png` now marks the Bonferroni-adjusted threshold |z| = 2.64 (α = 0.05/6 = 0.0083), the level the manuscript actually tests at. v15 drew the unadjusted |z| = 1.96.
3. **The figure-rename step now overwrites its targets.** v15 skipped any file that already existed, so stale PNGs from earlier runs survived into the LaTeX build folder — this is what caused figures to disagree with tables in earlier drafts.
4. The per-regime table prints all seven models instead of four, and checks that each row pools back to the full-sample RMSE. The computation always covered all seven; only the display was truncated.
5. The final summary computes its numbers rather than hard-coding them, and reports the resolution of the PT test (SE(DA) ≈ 1.85 pp, so the design detects DA above roughly 53.6% at α = 0.05).
6. Phase 6 flagged as not reported, with the reason stated inline.
7. New Phase 12 verification cell.
8. All stored outputs stripped, so nothing stale ships with the notebook.

## Note on `results/`

The CSVs here are from the v15 execution and are provided for reference; running the notebook regenerates all of them. Four files from that run are not included in this copy and should be added before archiving: `table7\_main\_results.csv`, `table15\_regimes.csv`, `table11\_conformal.csv`, `btc\_google\_trends\_daily.csv`. Also update `manifest\_v15.json`, whose `phase\_9\_b1\_onchain` and `phase\_10\_b4\_daily\_trends` flags read `false` although both phases ran — re-running Phase 11 regenerates it correctly.

## Citation

Bibliography and manuscript source are in `paper/`. Please cite the published article once it appears; until then, cite this repository and its Zenodo DOI.

## License

Suggested: MIT for the code, CC-BY-4.0 for the data and figures. Add a `LICENSE` file before publishing — Zenodo will ask for one.

