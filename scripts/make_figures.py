"""Figures 1-3 of the manuscript, drawn from the files saved by the v19 notebook.

Usage:  python make_figures.py [ROOT] [OUT]
    ROOT  repository root, with data/ and results/v19/ (default: parent of this script's folder)
    OUT   output folder (default: ROOT/figures/paper)

The script recomputes nothing new: it reads the Monte Carlo and positive-control tables (Fig. 1),
the test-period forecasts saved by the notebook (Fig. 2) and the monthly accuracies (Fig. 3), and
checks what it plots against the result tables before drawing. Fig. 2 needs data/ and
results/v19/_local/, which hold prices and are not redistributed: rebuild the dataset and run the
notebook first.

Format (Springer artwork guidelines): 174 mm wide, sans-serif lettering of 9-11 pt (Arial when it
is installed, otherwise a metric-compatible font), embedded TrueType fonts, parts denoted by
lowercase letters, no title inside the figure, and series told apart by line style and marker
shape as well as by colour.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
OUT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT / 'figures' / 'paper'
RES = ROOT / 'results' / 'v19'
OUT.mkdir(parents=True, exist_ok=True)

W = 174 / 25.4                       # inches: full width of a two-column Springer page
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'  # checked for colour-vision deficiency on white
INK, MUTED, GRID = '#0b0b0b', '#6f6e69', '#e4e3dd'
DASH = (0, (4, 2))
FAMILIES = ['RF', 'XGB', 'CNN-LSTM', 'GRU-Att']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'mathtext.fontset': 'custom', 'mathtext.rm': 'sans', 'mathtext.it': 'sans:italic',
    'mathtext.bf': 'sans:bold', 'mathtext.cal': 'sans:italic', 'mathtext.sf': 'sans', 'mathtext.tt': 'sans',
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'legend.frameon': False, 'legend.handlelength': 2.6,
    'figure.dpi': 100, 'savefig.dpi': 600,
})


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png')
    plt.close(fig)
    print(f'wrote {OUT / name}.pdf and .png')


def panel_letter(ax, letter):
    ax.set_title(letter, loc='left', fontweight='bold')


def year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))


def theory_da(rho, s):
    """Proposition 2: accuracy of the traded signal and accuracy on price-forecast changes, in %."""
    c = rho * s / np.sqrt(1 - 2 * rho * s + 2 * s ** 2)
    return 50 + 100 * np.arcsin(rho) / np.pi, 50 + 100 * np.arcsin(c) / np.pi


# ---------------------------------------------------------------------------------- inputs
t_mc = pd.read_csv(RES / 't09_monte_carlo.csv')
t_pcb = pd.read_csv(RES / 't11b_positive_control_resampled.csv')
t_month = pd.read_csv(RES / 't14c_monthly_changes.csv')
t_track = pd.read_csv(RES / 't14d_monthly_tracking.csv').set_index('model')
t_three = pd.read_csv(RES / 't14_three_measures.csv').set_index('forecast')
splits = pd.read_csv(RES / 't01_splits.csv').set_index('split')
T = int(splits.loc['test', 'days'])
high = splits.loc[['train', 'validation'], 'max close ($)'].max()

df = pd.read_csv(ROOT / 'data' / 'btc_features_2017-01-01_2026-11-12_no_trends.csv',
                 parse_dates=['date']).set_index('date').sort_index()
fc = pd.read_csv(RES / '_local' / 'forecasts_test_price.csv', parse_dates=[0], index_col=0)
test = df.loc[fc.index]
dates = fc.index
P_t, P_t1 = test['close'].values, test['target_price'].values
assert len(fc) == T and np.allclose(P_t, fc['Random Walk'].values)

SIG, CHG, RETC = 'signal', 'changes of prices', 'changes of returns'

# ---------------------------------------------------------------------------------- checks
cal = t_mc[t_mc['forecaster'].isin(['calibrated', 'no skill'])].sort_values('rho')
th = np.array([theory_da(r, s) for r, s in zip(t_mc['rho'], t_mc['s'])])
assert np.allclose(th[:, 0], t_mc[f'DA {SIG}, theory (%)']) and np.allclose(th[:, 1], t_mc[f'DA {CHG}, theory (%)'])
# the monthly accuracies, weighted by the days of each month, give back the accuracies of the test period
for name in FAMILIES:
    for target in ('price', 'return'):
        m = f'{name}-{target}'
        nxt, yhat = np.sign(np.diff(P_t1)), np.sign(np.diff(fc[m].values))
        keep = yhat != 0
        assert np.isclose(100 * np.mean(nxt[keep] == yhat[keep]), t_three.loc[m, 'DA changes of prices (%)']), m
        month = pd.PeriodIndex(dates[1:], freq='M').astype(str)
        by_month = pd.Series(nxt[keep] == yhat[keep]).groupby(month[keep]).mean() * 100
        assert np.allclose(by_month.reindex(t_month['month']).values, t_month[m].values), m
        assert np.isclose(np.corrcoef(t_month[m], t_month['repeat share (%)'])[0, 1],
                          t_track.loc[m, 'corr of monthly DA on changes of prices with the monthly repeat share']), m
print(f'checks passed: theory matches t09; monthly accuracies match the forecasts and t14; T = {T}')

# ---------------------------------------------------------------------------------- Fig. 1
grid = np.linspace(0, 0.5, 101)
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9), layout='constrained')
a.plot(grid, [theory_da(r, r)[0] for r in grid], color=BLUE, lw=1.4)
a.plot(grid, [theory_da(r, r)[1] for r in grid], color=ORANGE, lw=1.4, ls=DASH)
a.plot(cal['rho'], cal[f'DA {RETC}, simulated (%)'], '^', color=AQUA, ms=10.5, mew=0)
a.plot(cal['rho'], cal[f'DA {SIG}, simulated (%)'], 'o', color=BLUE, ms=4.2)
a.plot(cal['rho'], cal[f'DA {CHG}, simulated (%)'], 's', color=ORANGE, ms=4.5)
a.set_xlabel(r'Skill $\rho$')
a.set_ylabel('Directional accuracy (%)')
a.legend(handles=[Line2D([], [], color=BLUE, lw=1.4, marker='o', ms=4.2, label='Traded signal'),
                  Line2D([], [], color=ORANGE, lw=1.4, ls=DASH, marker='s', ms=4.5, label='Price-forecast changes'),
                  Line2D([], [], color=AQUA, lw=1.4, ls=(0, (1, 1.5)), marker='^', ms=8.5, mew=0, label='Return-forecast changes')],
         loc='upper left')
b.axhline(5, color=MUTED, lw=0.8, ls='-')
b.plot(cal['rho'], cal[f'PT rejections {RETC} (%)'], '^', ls=(0, (1, 1.5)), color=AQUA, ms=9, mew=0, lw=1.4)
b.plot(cal['rho'], cal[f'PT rejections {SIG} (%)'], 'o-', color=BLUE, ms=4.2, lw=1.4)
b.plot(cal['rho'], cal[f'PT rejections {CHG} (%)'], 's', ls=DASH, color=ORANGE, ms=4.5, lw=1.4)
b.plot(t_pcb['rho'], t_pcb[f'size-adjusted power {RETC} (%)'], '^', mfc='none', mec=AQUA, ms=10, mew=1.2)
b.plot(t_pcb['rho'], t_pcb[f'size-adjusted power {SIG} (%)'], 'o', mfc='none', mec=BLUE, ms=7.5, mew=1.2)
b.plot(t_pcb['rho'], t_pcb[f'size-adjusted power {CHG} (%)'], 's', mfc='none', mec=ORANGE, ms=7.5, mew=1.2)
b.set_xlabel(r'Skill $\rho$')
b.set_ylabel('Rejections of the PT test (%)')
b.legend(handles=[Line2D([], [], color=INK, lw=0, marker='o', ms=4.5, label='Gaussian returns'),
                  Line2D([], [], color=INK, lw=0, marker='o', ms=7.5, mew=1.2, mfc='none',
                         label='Resampled returns,\nsize-adjusted')],
         loc='lower right', bbox_to_anchor=(1.0, 0.08), handletextpad=0.2)
for ax, letter in ((a, 'a'), (b, 'b')):
    panel_letter(ax, letter)
    ax.set_xlim(-0.02, 0.52)
save(fig, 'Fig1')

# ---------------------------------------------------------------------------------- Fig. 2
fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True, layout='constrained')
for ax, letter, names in zip(axes, 'ab', (['RF-price', 'XGB-price'], ['CNN-LSTM-price', 'GRU-Att-price'])):
    ax.axhline(high, color=MUTED, lw=0.9, ls=':')
    ax.plot(dates, P_t1, color=INK, lw=1.1)
    ax.plot(dates, fc[names[0]], color=BLUE, lw=1.2)
    ax.plot(dates, fc[names[1]], color=ORANGE, lw=1.2, ls=DASH)
    # the dotted line (highest close of 2017-2023) is identified in the figure note
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.1, label='Close'),
                       Line2D([], [], color=BLUE, lw=1.2, label=names[0]),
                       Line2D([], [], color=ORANGE, lw=1.2, ls=DASH, label=names[1])],
              loc='upper left', borderaxespad=0.2)
    ax.set_ylim(35000, 137000)
    year_axis(ax)
    ax.set_xlim(dates[0], dates[-1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    panel_letter(ax, letter)
axes[0].set_ylabel('USD')
save(fig, 'Fig2')

# ---------------------------------------------------------------------------------- Fig. 3
# One panel per model: monthly accuracy on price-forecast changes against the monthly repeat share.
lim = (10, 90)
fig, axes = plt.subplots(2, 4, figsize=(W, 3.05), sharex=True, sharey=True, layout='constrained')
for row, (target, colour, marker) in enumerate((('return', BLUE, 'o'), ('price', ORANGE, 's'))):
    for col, fam in enumerate(FAMILIES):
        ax, m = axes[row, col], f'{fam}-{target}'
        ax.plot(lim, lim, color=MUTED, lw=0.8)
        ax.plot(t_month['repeat share (%)'], t_month[m], marker, color=colour, ms=3.6, mew=0, alpha=0.85)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xticks([20, 50, 80])
        ax.set_yticks([20, 50, 80])
        ax.grid(True, axis='both')
        r = t_track.loc[m, 'corr of monthly DA on changes of prices with the monthly repeat share']
        ax.text(0.04, 0.96, m, transform=ax.transAxes, va='top', ha='left', fontsize=9)
        ax.text(0.96, 0.05, f'r = {r:.2f}'.replace('-', '−'), transform=ax.transAxes, va='bottom', ha='right', fontsize=9)
    panel_letter(axes[row, 0], 'ab'[row])
fig.supxlabel('Repeat share of the realised direction in the month (%)', fontsize=10)
fig.supylabel('Accuracy on price-forecast\nchanges in the month (%)', fontsize=10)
save(fig, 'Fig3')
