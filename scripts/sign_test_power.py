"""Power of the exact sign test and its p-values under other references, from the counts saved by the notebook.

Usage: python scripts/sign_test_power.py results/v19

Reads t05_direction.csv and prints, for RF-price and XGB-price:
- the p-value of the exact sign test over all test days (number of correct days binomial with parameter one half);
- its power at the 5% level against a signal that is right on its long days as often as the model and on one
  half of its short days;
- the p-values of the same counts if a rise had a probability of 0.51 or 0.49 on every day;
- the share of rises among the test days.
The number of correct days is the sum of two independent binomial counts, one over the long days and one over
the short days, whose distribution is obtained by convolution.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def count_pmf(n_long, p_long, n_short, p_short):
    """Distribution of the number of correct days: long days correct with p_long, short days with p_short."""
    a = stats.binom.pmf(np.arange(n_long + 1), n_long, p_long)
    b = stats.binom.pmf(np.arange(n_short + 1), n_short, p_short)
    return np.convolve(a, b)


def main(res):
    dr = pd.read_csv(Path(res) / 't05_direction.csv').set_index('model')
    for m in ('RF-price', 'XGB-price'):
        r = dr.loc[m]
        n_long, n_short = int(r['long days']), int(r['short days'])
        hits, up_long = int(r['correct days signal']), int(r['up days among the long days'])
        down_short = int(r['down days among the short days'])
        T = n_long + n_short
        crit = next(c for c in range(T + 2) if stats.binom.sf(c - 1, T, 0.5) <= 0.05)
        p_half = count_pmf(n_long, 0.5, n_short, 0.5)[hits:].sum()
        assert abs(p_half - r['sign test p signal']) < 1e-9, 'sign test p-value differs from the notebook'
        power = count_pmf(n_long, up_long / n_long, n_short, 0.5)[crit:].sum()
        p51 = count_pmf(n_long, 0.51, n_short, 0.49)[hits:].sum()
        p49 = count_pmf(n_long, 0.49, n_short, 0.51)[hits:].sum()
        rises = up_long + n_short - down_short
        print(f'{m}: {hits} correct days of {T} (long {n_long}, of which {up_long} followed by a rise); '
              f'sign test p = {p_half:.4f}; power at 5% (critical count {crit}) = {100 * power:.1f}%; '
              f'p if P(rise) = 0.51: {p51:.4f}; if 0.49: {p49:.4f}; rises: {100 * rises / T:.1f}% of days')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/v19')
