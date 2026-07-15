"""The writing-and-plotting layer: one table and one figure per run.

The writer and the plotter live in one file because both are peripheral
to the physics: one writes an already computed table, the other draws it.
Every problem kind writes one tab-separated table into the output
directory and one matching figure under figures/, both under the same
base name, using only values the production path already computes and the
test suite already checks, so a figure carries no numerical claim beyond
what is already trusted.
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .solvers import (
    exact_avalanche_size_distribution,
    exact_mfpt,
    exact_survival,
    kmc_avalanche_sizes,
    kmc_mfpt,
    kmc_survival,
)
from .utils import ProblemInstance

PLOT_DEFAULTS = {
    "figsize": (7.0, 4.5),
    "dpi": 150,
    "closed_form_colour": "#1a1a1a",
    "exact_colour": "#2166ac",
    "kmc_colour": "#b2182b",
    "tt_colour": "#1b7837",
    "guide_colour": "#888888",
}


def run_tag(instance: ProblemInstance) -> str:
    """Base name shared by the table and the figure of one run."""
    return f"{instance.kind}_{instance.name}"


def flag_tag(kind: str, nbits: int, rate_up: float, rate_down: float, start: int,
             sites: int, grains: int, tmax: float, tpoints: int, smax: int,
             nshots: int, seed: int) -> str:
    """Deterministic output tag for one flag-driven parameter combination.

    The tag carries only the fields the given kind reads, so 2
    combinations that differ in any one of them never collide on the
    output filename. Kept in the library, not the entry file, so the tag
    rule is unit-tested directly.
    """
    if kind == "barrier_survival":
        return f"flag_nbits{nbits}_up{rate_up}_down{rate_down}_start{start}_tmax{tmax}_tpoints{tpoints}_nshots{nshots}_seed{seed}"
    if kind == "barrier_mfpt":
        return f"flag_nbits{nbits}_up{rate_up}_down{rate_down}_start{start}_nshots{nshots}_seed{seed}"
    if kind == "manna_avalanche":
        return f"flag_sites{sites}_grains{grains}_smax{smax}_nshots{nshots}_seed{seed}"
    if kind == "manna_survival":
        return f"flag_sites{sites}_grains{grains}_tmax{tmax}_tpoints{tpoints}_nshots{nshots}_seed{seed}"
    raise ValueError(f"unknown kind: {kind}")


def write_table(path: pathlib.Path, header: list[str], rows: list[list]) -> None:
    """Write one tab-separated table with a header line."""
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare(instance: ProblemInstance, outdir: str | pathlib.Path):
    outdir = pathlib.Path(outdir)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    tag = run_tag(instance)
    return outdir / f"{tag}.tsv", figdir / f"{tag}.png"


def run_problem(instance: ProblemInstance, outdir: str | pathlib.Path | None = None):
    """Run one problem instance: compute, write the table, draw the figure.

    Returns
    -------
    namespace with fields table and figure
        The two files the run produced.
    """
    outdir = instance.output if outdir is None else outdir
    table_path, figure_path = _prepare(instance, outdir)
    if instance.kind == "barrier_survival":
        _run_barrier_survival(instance, table_path, figure_path)
    elif instance.kind == "barrier_mfpt":
        _run_barrier_mfpt(instance, table_path, figure_path)
    elif instance.kind == "manna_avalanche":
        _run_manna_avalanche(instance, table_path, figure_path)
    elif instance.kind == "manna_survival":
        _run_manna_survival(instance, table_path, figure_path)
    else:
        raise ValueError(f"unknown kind: {instance.kind}")

    class Destination:
        table = table_path
        figure = figure_path

    return Destination


def _run_barrier_survival(instance, table_path, figure_path):
    from .models import firstpassage as fp
    walk = instance.model()
    times = np.linspace(0.0, instance.tmax, instance.tpoints)
    closed = walk.analytic_survival(times)
    columns = {"time": times, "closed_form": closed}
    if "exact" in instance.engines:
        columns["exact"] = exact_survival(walk, times)
    if "kmc" in instance.engines:
        columns["kmc"] = kmc_survival(walk, times, instance.nshots, instance.seed)
    if "tt" in instance.engines:
        columns["tt"] = fp.tt_survival(walk, times, cutoff=instance.cutoff,
                                       maxdim=instance.maxdim)
    header = list(columns)
    rows = [[f"{columns[c][i]:.12g}" for c in header] for i in range(len(times))]
    write_table(table_path, header, rows)

    fig, ax = plt.subplots(figsize=PLOT_DEFAULTS["figsize"], dpi=PLOT_DEFAULTS["dpi"])
    ax.plot(times, closed, "-", color=PLOT_DEFAULTS["closed_form_colour"],
            label="closed form", zorder=1)
    markers = {"exact": ("o", "exact_colour"), "kmc": ("s", "kmc_colour"), "tt": ("^", "tt_colour")}
    for name, (marker, colour) in markers.items():
        if name in columns:
            ax.plot(times, columns[name], marker, ms=5, mfc="none",
                    color=PLOT_DEFAULTS[colour], label=name, zorder=2)
    if "tt" in columns:
        deviation = np.max(np.abs(columns["tt"] - closed))
        ax.set_title(f"survival, largest tensor-train deviation {deviation:.1e}")
    ax.set_xlabel("time")
    ax.set_ylabel("survival probability")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def _run_barrier_mfpt(instance, table_path, figure_path):
    from .models import firstpassage as fp
    walk = instance.model()
    closed = walk.analytic_mfpt()
    rows = [["closed_form", f"{closed:.12g}", "0"]]
    values = {}
    if "exact" in instance.engines:
        values["exact"] = (exact_mfpt(walk), 0.0)
    if "kmc" in instance.engines:
        values["kmc"] = kmc_mfpt(walk, instance.nshots, instance.seed)
    if "tt" in instance.engines:
        mfpt, residual = fp.tt_mfpt(walk, cutoff=instance.cutoff, maxdim=instance.maxdim)
        values["tt"] = (mfpt, residual)
    for name, (value, uncertainty) in values.items():
        rows.append([name, f"{value:.12g}", f"{uncertainty:.3g}"])
    write_table(table_path, ["engine", "mfpt", "error_or_residual"], rows)

    fig, ax = plt.subplots(figsize=PLOT_DEFAULTS["figsize"], dpi=PLOT_DEFAULTS["dpi"])
    names = list(values)
    heights = [values[n][0] for n in names]
    colours = [PLOT_DEFAULTS[f"{n}_colour"] for n in names]
    ax.bar(names, heights, color=colours)
    if "kmc" in values:
        k = names.index("kmc")
        ax.errorbar([k], [values["kmc"][0]], yerr=[values["kmc"][1]],
                    fmt="none", ecolor="black", capsize=4)
    ax.axhline(closed, color=PLOT_DEFAULTS["closed_form_colour"], linestyle="--",
               label=f"closed form {closed:.4f}")
    ax.set_ylabel("mean first-passage time")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def _run_manna_avalanche(instance, table_path, figure_path):
    from .models import manna as mn
    pile = instance.model()
    smax = instance.smax
    sizes = np.arange(smax + 1)
    columns = {"size": sizes}
    if "exact" in instance.engines:
        columns["exact"] = exact_avalanche_size_distribution(pile, smax)
    if "kmc" in instance.engines:
        samples = kmc_avalanche_sizes(pile, instance.nshots, instance.seed)
        histogram = np.bincount(samples[samples <= smax], minlength=smax + 1)
        columns["kmc"] = histogram / len(samples)
    if "tt" in instance.engines:
        columns["tt"] = mn.tt_avalanche_size_distribution(
            pile, smax, cutoff=instance.cutoff, maxdim=instance.maxdim)
    header = list(columns)
    rows = [[f"{columns[c][i]:.12g}" for c in header] for i in range(smax + 1)]
    write_table(table_path, header, rows)

    fig, ax = plt.subplots(figsize=PLOT_DEFAULTS["figsize"], dpi=PLOT_DEFAULTS["dpi"])
    positive = sizes >= 1
    if "exact" in columns:
        mask = positive & (columns["exact"] > 1e-300)
        ax.plot(sizes[mask], columns["exact"][mask], "-",
                color=PLOT_DEFAULTS["exact_colour"], label="exact embedded chain")
    if "kmc" in columns:
        mask = positive & (columns["kmc"] > 0)
        ax.plot(sizes[mask], columns["kmc"][mask], "s", ms=4, mfc="none",
                color=PLOT_DEFAULTS["kmc_colour"], label=f"kinetic Monte Carlo, {instance.nshots} runs")
    if "tt" in columns:
        mask = positive & (columns["tt"] > 1e-14)
        ax.plot(sizes[mask], columns["tt"][mask], "^", ms=4, mfc="none",
                color=PLOT_DEFAULTS["tt_colour"], label="tensor train")
    anchor = None
    for name in ("exact", "tt", "kmc"):
        if name in columns and columns[name][1] > 0:
            anchor = columns[name][1]
            break
    if anchor is not None:
        guide = anchor * (sizes[positive] / 1.0) ** (-1.5)
        ax.plot(sizes[positive], guide, ":", color=PLOT_DEFAULTS["guide_colour"],
                label="mean-field exponent guide, size**(-3/2)")
    ax.set_xlabel("avalanche size (topplings)")
    ax.set_ylabel("probability")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def _run_manna_survival(instance, table_path, figure_path):
    from .models import manna as mn
    pile = instance.model()
    times = np.linspace(0.0, instance.tmax, instance.tpoints)
    columns = {"time": times}
    if "exact" in instance.engines:
        columns["exact"] = exact_survival(pile, times)
    if "kmc" in instance.engines:
        columns["kmc"] = kmc_survival(pile, times, instance.nshots, instance.seed)
    if "tt" in instance.engines:
        columns["tt"] = mn.tt_survival(pile, times, cutoff=instance.cutoff,
                                       maxdim=instance.maxdim)
    header = list(columns)
    rows = [[f"{columns[c][i]:.12g}" for c in header] for i in range(len(times))]
    write_table(table_path, header, rows)

    fig, ax = plt.subplots(figsize=PLOT_DEFAULTS["figsize"], dpi=PLOT_DEFAULTS["dpi"])
    styles = {"exact": ("-", "exact_colour"), "kmc": ("s", "kmc_colour"), "tt": ("^", "tt_colour")}
    for name, (style, colour) in styles.items():
        if name in columns:
            ax.plot(times, columns[name], style, ms=4, mfc="none",
                    color=PLOT_DEFAULTS[colour], label=name)
    ax.set_xlabel("time")
    ax.set_ylabel("probability the pile is still active")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
