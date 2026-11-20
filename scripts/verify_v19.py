"""Independent recomputation of headline v19 numbers from the saved forecasts.

Uses its own code (statsmodels HAC for DM and CW, direct formulas elsewhere) and compares
with the tables written by the notebook.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent   # repository root
res = root / 'results' / 'v19'
df = pd.read_csv(root / 'data' / 'btc_features_2017-01-01_2026-11-12_no_trends.csv',
                 parse_dates=['date']).set_index('date')
fc = pd.read_csv(res / '_local' / 'forecasts_test_price.csv', parse_dates=[0], index_col=0, float_precision='round_trip')
test = df.loc[fc.index]
P0, P1 = test['close'].values, test['target_price'].values
r = np.log(P1 / P0)
T = len(r)
lag = int(np.floor(4 * (T / 100) ** (2 / 9)))

tb = pd.read_csv(root / 'raw' / 'fred_tb3ms.csv', parse_dates=['observation_date'])
tb['m'] = tb['observation_date'].dt.to_period('M')
rate = dict(zip(tb['m'], tb['TB3MS']))
last = tb['m'].max()
rf = np.array([(1 + rate[min(d.to_period('M'), last)] / 100) ** (1 / 365) - 1 for d in fc.index])
R = P1 / P0 - 1

acc = pd.read_csv(res / 't04_accuracy.csv').set_index('model')
dirn = pd.read_csv(res / 't05_direction.csv').set_index('model')
trd = pd.read_csv(res / 't06_trading.csv')

def hac_t(y):
    fit = sm.OLS(y, np.ones(len(y))).fit(cov_type='HAC', cov_kwds={'maxlags': lag, 'use_correction': False})
    return float(fit.tvalues[0])

TIE = 1e-12                # two consecutive forecasts closer than this are equal (as in the notebook)


def sign_change(x):
    d = np.diff(x)
    return np.where(np.abs(d) < TIE, 0.0, np.sign(d))


checks = []
for m in fc.columns:
    y = fc[m].values
    rh = np.log(y / P0)
    mine = {'RMSE ($)': np.sqrt(np.mean((P1 - y) ** 2)),
            'OOS R2 (%)': 100 * (1 - np.sum((r - rh) ** 2) / np.sum(r ** 2))}
    if m != 'Random Walk':
        d = (r - rh) ** 2 - r ** 2
        mine['DM'] = hac_t(d) * np.sqrt((T - 1) / T)
        mine['CW'] = hac_t(2 * r * rh)
    for k, v in mine.items():
        checks.append(('t04', m, k, v, acc.loc[m, k]))
    s = np.sign(y - P0)
    if m != 'Random Walk':
        keep = s != 0
        checks.append(('t05', m, 'DA signal (%)', 100 * np.mean(s[keep] == np.sign(P1 - P0)[keep]),
                       dirn.loc[m, 'DA signal (%)']))
    ch = sign_change(np.log(y))
    keep = ch != 0
    checks.append(('t05', m, 'DA changes (%)', 100 * np.mean(ch[keep] == np.sign(np.diff(P1))[keep]),
                   dirn.loc[m, 'DA changes (%)']))
    if m == 'Random Walk':
        continue
    for cost in (0, 50):
        pos = np.concatenate([[0.0], s])
        c = cost / 1e4 * np.abs(np.diff(pos))
        x = s * (R - rf) - c
        sr = np.sqrt(365) * x.mean() / x.std(ddof=1)
        cum = 100 * (np.prod(1 + s * R - c) - 1)
        row = trd[(trd['model'] == m) & (trd['cost (bp)'] == cost)].iloc[0]
        checks.append(('t06', m, f'Sharpe {cost}bp', sr, row['Sharpe']))
        checks.append(('t06', m, f'cumulative {cost}bp', cum, row['cumulative return (%)']))

fl = pd.read_csv(res / '_local' / 'forecasts_test_price_dl_long.csv', parse_dates=[0], index_col=0, float_precision='round_trip')
ep = pd.read_csv(res / 't10_dl_epochs.csv')
for m in fl.columns:
    for k, y in enumerate((fc[m].values, fl[m].values)):
        row = ep[ep['model'] == m].iloc[k]
        tag = f"{m} {'early stop' if k == 0 else 'long'}"
        rh = np.log(y / P0)
        checks.append(('t10', tag, 'RMSE ($)', np.sqrt(np.mean((P1 - y) ** 2)), row['RMSE ($)']))
        checks.append(('t10', tag, 'OOS R2 (%)', 100 * (1 - np.sum((r - rh) ** 2) / np.sum(r ** 2)), row['OOS R2 (%)']))
        checks.append(('t10', tag, 'DM', hac_t((r - rh) ** 2 - r ** 2) * np.sqrt((T - 1) / T), row['DM']))
        s = np.sign(y - P0)
        keep = s != 0
        checks.append(('t10', tag, 'DA signal (%)', 100 * np.mean(s[keep] == np.sign(P1 - P0)[keep]), row['DA signal (%)']))
        pos = np.concatenate([[0.0], s])
        x = s * (R - rf) - 50 / 1e4 * np.abs(np.diff(pos))
        checks.append(('t10', tag, 'Sharpe 50bp', np.sqrt(365) * x.mean() / x.std(ddof=1), row['Sharpe at 50 bp']))

# ---- direction tests, written independently of the notebook --------------------------------------
from scipy import stats as _st


def pt_p(actual_sign, forecast_sign):
    """Pesaran-Timmermann (1992) one-sided p-value; days with a zero sign are dropped."""
    k = (actual_sign != 0) & (forecast_sign != 0)
    y, x = (actual_sign[k] > 0).astype(float), (forecast_sign[k] > 0).astype(float)
    n = len(y)
    p_hat = np.mean(y == x)
    py, px = y.mean(), x.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v_p = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * px * (1 - px) + (2 * px - 1) ** 2 * py * (1 - py)) / n \
        + 4 * py * px * (1 - py) * (1 - px) / n ** 2
    return 100 * p_hat, float(_st.norm.sf((p_hat - p_star) / np.sqrt(v_p - v_star)))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


dP = np.sign(P1 - P0)
learned = [m for m in fc.columns if m != 'Random Walk']
p_chg, p_sig = [], []
for m in learned:
    y = fc[m].values
    da_s, p_s = pt_p(dP, np.sign(y - P0))
    da_c, p_c = pt_p(np.sign(np.diff(P1)), sign_change(np.log(y)))
    p_sig.append(p_s)
    p_chg.append(p_c)
    checks.append(('t05', m, 'PT p signal', p_s, dirn.loc[m, 'PT p signal']))
    checks.append(('t05', m, 'PT p changes', p_c, dirn.loc[m, 'PT p changes']))
    s_ = np.sign(y - P0)
    a_, f_ = dP > 0, s_ > 0
    table = [[int(np.sum(f_ & a_)), int(np.sum(f_ & ~a_))], [int(np.sum(~f_ & a_)), int(np.sum(~f_ & ~a_))]]
    checks.append(('t05', m, 'Fisher p signal', _st.fisher_exact(table, alternative='greater')[1],
                   dirn.loc[m, 'Fisher p signal']))
    rh = np.log(y / P0)
    checks.append(('t05', m, 'AC(1) of implied return', np.corrcoef(rh[1:], rh[:-1])[0, 1],
                   dirn.loc[m, 'AC(1) of implied return']))
    checks.append(('t05', m, 'mean implied return (%)', 100 * rh.mean(), dirn.loc[m, 'mean implied return (%)']))
adj = holm(p_chg + p_sig)
for j, m in enumerate(learned):
    checks.append(('t05', m, 'PT p changes, Holm (16)', adj[j], dirn.loc[m, 'PT p changes, Holm (16)']))
    checks.append(('t05', m, 'PT p signal, Holm (16)', adj[len(learned) + j], dirn.loc[m, 'PT p signal, Holm (16)']))

# level rules of t05c
rule = pd.read_csv(res / 't05c_level_rule.csv')
cap = df.loc[:'2023-12-31', 'close'].max()
for i, L in enumerate([cap, fc['XGB-price'].max(), fc['RF-price'].max()]):
    da, p = pt_p(dP, np.where(P0 < L, 1.0, -1.0))
    checks.append(('t05c', f'rule {i}', 'level ($)', L, rule.loc[i, 'level ($)']))
    checks.append(('t05c', f'rule {i}', 'DA (%)', da, rule.loc[i, 'DA (%)']))
    checks.append(('t05c', f'rule {i}', 'PT p', p, rule.loc[i, 'PT p']))
below = P0 < cap
for i, m in [(3, 'RF-price'), (4, 'XGB-price')]:
    da, p = pt_p(dP[below], np.sign(fc[m].values[below] - P0[below]))
    checks.append(('t05c', f'{m} below the high', 'DA (%)', da, rule.loc[i, 'DA (%)']))
    checks.append(('t05c', f'{m} below the high', 'PT p', p, rule.loc[i, 'PT p']))

# ---- the three measures (t14), written independently of the notebook ------------------------------------
def pt_stat(actual_sign, forecast_sign):
    """Pesaran-Timmermann statistic; days with a zero sign are dropped."""
    k = (actual_sign != 0) & (forecast_sign != 0)
    y, x = (actual_sign[k] > 0).astype(float), (forecast_sign[k] > 0).astype(float)
    n = len(y)
    p_hat, py, px = np.mean(y == x), y.mean(), x.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v = p_star * (1 - p_star) / n - (((2 * py - 1) ** 2 * px * (1 - px) + (2 * px - 1) ** 2 * py * (1 - py)) / n
                                     + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    return 100 * p_hat, (p_hat - p_star) / np.sqrt(v)


close = df['close'].values
i0 = df.index.get_loc(fc.index[0])
prev = close[i0 - 1:i0 - 1 + T]                       # P_{t-1} on each test day
r_last = np.log(P0 / prev)                            # last realised return
three = pd.read_csv(res / 't14_three_measures.csv').set_index('forecast')
no_model = {'Constant price': np.full(T, close[i0 - 1]),
            'Lagged price': prev,
            'Momentum': P0 ** 2 / prev,
            'Moving average, 7 days': np.array([close[i0 + j - 6:i0 + j + 1].mean() for j in range(T)]),
            'Moving average, 30 days': np.array([close[i0 + j - 29:i0 + j + 1].mean() for j in range(T)])}
for m, y in {**{c: fc[c].values for c in fc.columns}, **no_model}.items():
    rh = np.log(y / P0)
    fchg = sign_change(np.log(y))
    if (fchg != 0).any():
        k = fchg != 0
        same = fchg[k] == np.sign(r_last[1:])[k]
        rep_ = np.sign(r_last[1:])[k] == dP[1:][k]
        checks.append(('t14', m, 'lambda (%)', 100 * same.mean(), three.loc[m, 'forecast change in the direction of the last return (%)']))
        da_p = 100 * np.mean(fchg[k] == dP[1:][k])
        checks.append(('t14', m, 'DA changes of prices (%)', da_p, three.loc[m, 'DA changes of prices (%)']))
        # Proposition 1: DA = lambda * Q_A + (1 - lambda) * (1 - Q_notA)
        q_a = rep_[same].mean() if same.any() else 0.0
        q_b = rep_[~same].mean() if (~same).any() else 0.0
        checks.append(('t14', m, 'Proposition 1 identity (%)', 100 * (same.mean() * q_a + (1 - same.mean()) * (1 - q_b)), da_p))
        assert np.all(np.abs(np.diff(rh))[k][~same] >= np.abs(r_last[1:])[k][~same])
    d_hat, d_act = sign_change(rh), np.sign(np.diff(r))
    if (d_hat != 0).any():
        da_r, z_r = pt_stat(d_act, d_hat)
        checks.append(('t14', m, 'DA changes of returns (%)', da_r, three.loc[m, 'DA changes of returns (%)']))
        checks.append(('t14', m, 'PT z changes of returns', z_r, three.loc[m, 'PT z changes of returns']))
    if m not in fc.columns:
        s_ = np.sign(y - P0)
        checks.append(('t14', m, 'DA signal (%)', 100 * np.mean(s_ == dP), three.loc[m, 'DA signal (%)']))

# ---- forecasts without a model by calendar year (t14b): they need the price series only ---------------------
yrs = pd.read_csv(res / 't14b_no_model_by_year.csv').set_index('period')
lp = np.log(close)
ret_next = np.log(df['target_price'].values / close)          # r_{t+1} at each forecast date
for y_ in range(2018, 2027):
    idx = np.flatnonzero(df.index.year == y_)
    nxt, last_ = np.sign(ret_next[idx]), np.sign(lp[idx] - lp[idx - 1])
    checks.append(('t14b', str(y_), 'repeat share (%)', 100 * np.mean(nxt == last_), yrs.loc[str(y_), 'repeat share (%)']))
    d_next = np.sign(ret_next[idx] - ret_next[idx - 1])
    checks.append(('t14b', str(y_), 'constant price (%)', 100 * np.mean(-last_ == d_next),
                   yrs.loc[str(y_), 'Constant price: DA changes of returns (%)']))
    d_last = np.sign(ret_next[idx - 1] - ret_next[idx - 2])
    checks.append(('t14b', str(y_), 'lagged price (%)', 100 * np.mean(-d_last == d_next),
                   yrs.loc[str(y_), 'Lagged price: DA changes of returns (%)']))

# ---- monthly tracking (t14d) and correlations (t15) -------------------------------------------------------------
track = pd.read_csv(res / 't14d_monthly_tracking.csv').set_index('model')
corr_t = pd.read_csv(res / 't15_correlation.csv').set_index('model')
month = pd.PeriodIndex(fc.index[1:], freq='M').astype(str)
rep_all = pd.Series(np.sign(r_last[1:]) == dP[1:]).groupby(month).mean()
big10 = np.argsort(-np.abs(r))[:10]
rest = np.setdiff1d(np.arange(T), big10)
for m in learned:
    y = fc[m].values
    fchg = sign_change(np.log(y))
    k = fchg != 0
    by_m = pd.Series(fchg[k] == dP[1:][k]).groupby(month[k]).mean()
    checks.append(('t14d', m, 'monthly correlation', np.corrcoef(by_m.values, rep_all.reindex(by_m.index).values)[0, 1],
                   track.loc[m, 'corr of monthly DA on changes of prices with the monthly repeat share']))
    rh = np.log(y / P0)
    checks.append(('t15', m, 'Pearson', np.corrcoef(rh, r)[0, 1], corr_t.loc[m, 'Pearson']))
    checks.append(('t15', m, 'Spearman', _st.spearmanr(rh, r)[0], corr_t.loc[m, 'Spearman']))
    checks.append(('t15', m, 'Pearson without ten days', np.corrcoef(rh[rest], r[rest])[0, 1],
                   corr_t.loc[m, 'Pearson without the 10 largest absolute returns']))

# ---- baselines and refits (t13, t13b) from the saved forecasts ---------------------------------------------------
fx = pd.read_csv(res / '_local' / 'forecasts_test_price_extra.csv', parse_dates=[0], index_col=0, float_precision='round_trip')
t13 = pd.concat([pd.read_csv(res / 't13_baselines_refits.csv'), pd.read_csv(res / 't13b_price_targets_monthly_refit.csv')]).set_index('model')
for m in fx.columns:
    if m not in t13.index:
        continue
    y = fx[m].values
    rh = np.log(y / P0)
    checks.append(('t13', m, 'OOS R2 (%)', 100 * (1 - np.sum((r - rh) ** 2) / np.sum(r ** 2)), t13.loc[m, 'OOS R2 (%)']))
    checks.append(('t13', m, 'DM', hac_t((r - rh) ** 2 - r ** 2) * np.sqrt((T - 1) / T), t13.loc[m, 'DM']))
    s_ = np.sign(y - P0)
    checks.append(('t13', m, 'DA signal (%)', 100 * np.mean(s_[s_ != 0] == dP[s_ != 0]), t13.loc[m, 'DA signal (%)']))
    da_r, z_r = pt_stat(np.sign(np.diff(r)), sign_change(rh))
    checks.append(('t13', m, 'DA changes of returns (%)', da_r, t13.loc[m, 'DA changes of returns (%)']))
    checks.append(('t13', m, 'PT z changes of returns', z_r, t13.loc[m, 'PT z changes of returns']))

out = pd.DataFrame(checks, columns=['table', 'model', 'quantity', 'recomputed', 'notebook'])
out['abs diff'] = (out['recomputed'] - out['notebook']).abs()
out['rel diff'] = out['abs diff'] / out['notebook'].abs().replace(0, np.nan)
print(out.to_string(index=False, float_format=lambda v: f'{v:.6g}'))
# The saved forecasts are read with float_precision='round_trip', so they equal the values of the notebook,
# and two consecutive forecasts closer than TIE are treated as equal, as in the notebook.
# One exception: the forecasts of XGB-price are 32-bit numbers saved in their shortest decimal form, so a price
# level read from the file (the highest forecast) can differ from the value of the notebook by less than half a cent.
level = (out['quantity'] == 'level ($)') & (out['abs diff'] < 0.005)
bad = out[(out['rel diff'] > 1e-9) & (out['abs diff'] > 1e-9) & ~level]
print(f'\n{len(out)} quantities recomputed; {len(bad)} differ by more than 1e-9 (relative)')
if len(bad):
    print(bad.to_string(index=False))
