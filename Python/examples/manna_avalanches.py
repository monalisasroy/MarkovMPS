#!/usr/bin/env python3
"""Headline avalanche run: the committed Manna instances.

manna_small compares all three engines on a 4-site ring, including the
machine-precision tail of the exact embedded chain and the
Fourier-inverted tensor-train generating function; it takes seconds.
manna_critical runs kinetic Monte Carlo alone on a 24-site ring near the
critical density, the stated boundary of the package; it takes about a
minute. A tensor-train run at intermediate size (5 sites) costs a few
minutes in this NumPy implementation because the Fourier inversion solves
one linear system per point of the unit circle; the Julia tree is the
faster route at that size.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from markovmps.plot_utils import run_problem
from markovmps.utils import load_problem

for name in ("manna_small", "manna_critical"):
    instance = load_problem(ROOT / "problems" / f"{name}.toml")
    destination = run_problem(instance, outdir=ROOT / "results" / "python")
    print(f"wrote {destination.table} and {destination.figure}")
