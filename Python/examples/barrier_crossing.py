#!/usr/bin/env python3
"""Headline first-passage run: the committed barrier instances.

Runs the survival curve and the mean first-passage time of the biased walk
with all three engines and writes the tables and figures under results/. The
survival curve lands on the closed spectral form and the hitting time on
the gambler's ruin value; the largest tensor-train deviation is printed in
the figure title. Runtime is a few seconds.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from markovmps.plot_utils import run_problem
from markovmps.utils import load_problem

for name in ("barrier", "barrier_mfpt"):
    instance = load_problem(ROOT / "problems" / f"{name}.toml")
    destination = run_problem(instance, outdir=ROOT / "results" / "python")
    print(f"wrote {destination.table} and {destination.figure}")
