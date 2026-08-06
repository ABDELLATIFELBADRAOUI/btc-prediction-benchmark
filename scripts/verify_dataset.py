#!/usr/bin/env python3
"""
verify_dataset.py -- confirm that a reconstructed btc_features_final.csv is
identical to the one used to produce every number in the manuscript.

The input series are not redistributed: the data providers' terms of use do not
permit republication of their exports. Instead we publish this fingerprint, so
that anyone who repeats the acquisition step can confirm they have the same
dataset before running the benchmark.

Works from the shell or from a notebook cell.

    # shell
    python scripts/verify_dataset.py                       # finds data/ automatically
    python scripts/verify_dataset.py path/to/features.csv

    # notebook
    from verify_dataset import verify
    verify()                                               # or verify('path/to/features.csv')

`verify()` returns True/False and never raises SystemExit, so it is safe to call
in a cell. Run as a script, the exit status is 0 on success and 1 on failure.

The reference moments below are taken verbatim from
results/table1_descriptive_stats.csv, the file that produced Table 2 of the
manuscript.
"""

import sys
import hashlib
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Reference fingerprint
# --------------------------------------------------------------------------
N_ROWS = 2525
DATE_FIRST = "2017-01-31"
DATE_LAST = "2023-12-30"

# column: (mean, std, min, max) -- results/table1_descriptive_stats.csv
MOMENTS = {
    "close":         (19430.77226138614,     16109.508085185145,   937.52,               67566.83),
    "log_return":    (0.0015146008011317547, 0.03871235924469161, -0.4647297680590224,   0.2251189979873116),
    "volatility_7d": (0.03280239380795325,   0.020201869492714963, 0.0019983307404373,   0.1902841401882699),
    "rsi_14":        (53.44583973091962,     18.772168535004006,   4.699029586921967,    99.76562792965088),
    "google_trends": (24.253069306930694,    15.436214220922054,   6.0,                  100.0),
}

# google_trends comes from a provider that resamples on every download, so it is
# verifiable only to within that sampling variability. Price-derived columns
# must match to floating-point precision.
TOL_STRICT = 1e-6
TOL_TRENDS = 2e-2


def _default_path() -> Path:
    """Locate data/btc_features_final.csv from the script or the working dir."""
    anchors = []
    try:
        anchors += [Path(__file__).resolve().parent,
                    Path(__file__).resolve().parent.parent]
    except NameError:                       # pasted into a notebook cell
        pass
    here = Path.cwd().resolve()
    anchors += [here, *here.parents]
    seen = set()
    for base in anchors:
        if base in seen:
            continue
        seen.add(base)
        for candidate in (base / "data" / "btc_features_final.csv",
                          base / "btc_features_final.csv"):
            if candidate.exists():
                return candidate
    return Path("data/btc_features_final.csv")


def verify(path=None, verbose: bool = True) -> bool:
    """Check a feature file against the published fingerprint. Returns True if it matches."""
    path = _default_path() if path is None else Path(path)
    say = print if verbose else (lambda *a, **k: None)

    if not path.exists():
        say(f"File not found: {path}\n")
        say("Place btc_features_final.csv in data/, or pass its path:")
        say("    verify('path/to/btc_features_final.csv')")
        say("\nSee data/README.md for how to rebuild it from the primary sources.")
        return False

    failures = []

    def check(label, ok, detail=""):
        say(f"  {'OK  ' if ok else 'FAIL'}  {label}{('   ' + detail) if detail and not ok else ''}")
        if not ok:
            failures.append(label)

    say(f"Verifying {path}\n")
    say(f"SHA-256: {hashlib.sha256(path.read_bytes()).hexdigest()}")
    say("  (for your own records; it can differ from ours through line endings or\n"
        "   column order alone, which the checks below tolerate)\n")

    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as exc:
        say(f"Could not read the file as CSV with a 'date' column: {exc}")
        return False

    say("Shape and coverage")
    check(f"row count is {N_ROWS}", len(df) == N_ROWS, f"got {len(df)}")
    check(f"first date is {DATE_FIRST}",
          str(df["date"].min().date()) == DATE_FIRST, f"got {df['date'].min().date()}")
    check(f"last date is {DATE_LAST}",
          str(df["date"].max().date()) == DATE_LAST, f"got {df['date'].max().date()}")
    n_na = int(df.isna().sum().sum())
    check("no missing values", n_na == 0, f"got {n_na}")
    check("dates unique and sorted", df["date"].is_monotonic_increasing and df["date"].is_unique)

    say("\nSchema")
    for col in MOMENTS:
        check(f"column '{col}' present", col in df.columns)

    say("\nMoments of the input series")
    for col, (mu, sd, lo, hi) in MOMENTS.items():
        if col not in df.columns:
            continue
        tol = TOL_TRENDS if col == "google_trends" else TOL_STRICT
        s = df[col].astype(float)
        for name, ref, got in (("mean", mu, s.mean()), ("std", sd, s.std()),
                               ("min", lo, s.min()),   ("max", hi, s.max())):
            denom = abs(ref) if abs(ref) > 1e-12 else 1.0
            check(f"{col}.{name}", abs(got - ref) / denom <= tol,
                  f"expected {ref:.10g}, got {got:.10g}")

    say("\n" + "=" * 66)
    if failures:
        say(f"{len(failures)} CHECK(S) FAILED -- this is NOT the dataset used in the paper.")
        say("Numbers produced from it will not match the published tables.")
        if all(f.startswith("google_trends") for f in failures):
            say("\nOnly google_trends failed. That column comes from a provider that")
            say("resamples on each download, so a mismatch there is expected and")
            say("affects only the attention feature, not the price-derived results.")
    else:
        say("ALL CHECKS PASSED -- dataset matches the one used in the manuscript.")
    say("=" * 66)
    return not failures


def _running_in_notebook() -> bool:
    return "ipykernel" in sys.modules or "IPython" in sys.modules


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) > 1:
        print("Usage: python verify_dataset.py [path/to/btc_features_final.csv]")
        ok = False
    else:
        ok = verify(args[0] if args else None)
    # Only set a shell exit status when actually run as a script; inside a
    # notebook sys.exit() surfaces as a confusing SystemExit traceback.
    if not _running_in_notebook():
        sys.exit(0 if ok else 1)
