#!/usr/bin/env python3
"""Command-line entry point over the markovmps library.

The library holds the logic; this file resolves parameters and
dispatches. Named problem instances live in problems/ at the repository
root, and flag-driven
runs use the same ProblemInstance type. Every parameter flag accepts one
or more values; a flag given several values repeats the run for each
combination in one invocation, the same sweep convention as run.jl in the
Julia tree.

Usage:
  python3 run.py --problem NAME [NAME ...] [--output-dir PATH]
  python3 run.py [--kind KIND] [parameter flags] [--output-dir PATH]

Problem instances:
  python3 run.py --problem barrier
  python3 run.py --problem barrier_mfpt
  python3 run.py --problem manna_small
  python3 run.py --problem manna_critical
  python3 run.py --problem all

Flag run:
  --kind NAME          barrier_survival | barrier_mfpt | manna_avalanche | manna_survival
  --nbits INT [...]    binary sites of the biased walk          (default 4)
  --rate-up F [...]    upward hop rate                          (default 1.0)
  --rate-down F [...]  downward hop rate                        (default 1.0)
  --start INT [...]    starting position                        (default half range)
  --sites INT [...]    Manna ring size                          (default 6)
  --grains INT [...]   Manna grain count                        (default 4)
  --tmax F [...]       longest time on a survival curve         (default 10.0)
  --tpoints INT [...]  times on a survival curve                (default 21)
  --smax INT [...]     largest avalanche size resolved          (default 60)
  --nshots INT [...]   Monte Carlo trajectories                 (default 2000)
  --seed INT [...]     Monte Carlo seed                         (default 1)
  --engines E [...]    subset of exact kmc tt                   (default all 3)
  --output-dir PATH    directory for run files        (default results/python at the root)
  -h, --help           show this message and exit

Sweep: any parameter flag above takes one or more values; the run repeats
for every combination, writing one table and one figure per combination
under a tag built from the fields the kind reads, so no 2 combinations
collide on a filename.
"""

from __future__ import annotations

import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from markovmps.plot_utils import flag_tag, run_problem
from markovmps.utils import ProblemInstance, load_problem, problem_paths

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBLEM_DIR = ROOT / "problems"
DEFAULT_OUTDIR = ROOT / "results" / "python"


def parse_args(args: list[str]) -> dict[str, list[str]]:
    """Collect --flag value [value ...] pairs; a bare flag maps to true."""
    options: dict[str, list[str]] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("--"):
            name = token[2:]
            values: list[str] = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                values.append(args[i])
                i += 1
            options[name] = values if values else ["true"]
        else:
            i += 1
    return options


def get_values(options, name, default):
    """Every value given for a flag, or a one-element default list."""
    return options.get(name, [str(default)])


def main() -> None:
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        return
    options = parse_args(args)
    outdir = options.get("output-dir", [str(DEFAULT_OUTDIR)])[0]

    if "problem" in options:
        for path in problem_paths(options["problem"], PROBLEM_DIR):
            instance = load_problem(path)
            destination = run_problem(instance, outdir=outdir)
            print(f"wrote {destination.table} and {destination.figure}")
        return

    kind = options.get("kind", ["barrier_survival"])[0]
    engines = tuple(options.get("engines", ["exact", "kmc", "tt"]))
    nbits_values = [int(v) for v in get_values(options, "nbits", 4)]
    up_values = [float(v) for v in get_values(options, "rate-up", 1.0)]
    down_values = [float(v) for v in get_values(options, "rate-down", 1.0)]
    start_values = options.get("start")
    sites_values = [int(v) for v in get_values(options, "sites", 6)]
    grains_values = [int(v) for v in get_values(options, "grains", 4)]
    tmax_values = [float(v) for v in get_values(options, "tmax", 10.0)]
    tpoints_values = [int(v) for v in get_values(options, "tpoints", 21)]
    smax_values = [int(v) for v in get_values(options, "smax", 60)]
    nshots_values = [int(v) for v in get_values(options, "nshots", 2000)]
    seed_values = [int(v) for v in get_values(options, "seed", 1)]

    for combo in itertools.product(nbits_values, up_values, down_values,
                                   sites_values, grains_values, tmax_values,
                                   tpoints_values, smax_values, nshots_values,
                                   seed_values):
        nbits, up, down, sites, grains, tmax, tpoints, smax, nshots, seed = combo
        starts = [int(v) for v in start_values] if start_values else [2 ** (nbits - 1)]
        for start in starts:
            tag = flag_tag(kind, nbits, up, down, start, sites, grains,
                           tmax, tpoints, smax, nshots, seed)
            instance = ProblemInstance(
                name=tag, kind=kind, output=outdir,
                nbits=nbits, rate_up=up, rate_down=down, start=start,
                sites=sites, grains=grains, nmax=grains, seed_site=sites // 2,
                tmax=tmax, tpoints=tpoints, smax=smax, nshots=nshots,
                seed=seed, maxdim=64, cutoff=1e-10, engines=engines,
            )
            destination = run_problem(instance, outdir=outdir)
            print(f"wrote {destination.table} and {destination.figure}")


if __name__ == "__main__":
    main()
