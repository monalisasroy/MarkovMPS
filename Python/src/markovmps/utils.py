"""Problem instances: the input boundary.

A committed TOML file names a run completely: the model, its parameters,
the observable, and the engines to compare. Validation happens here, at
the boundary where data enters, and nowhere else: an unrecognised key is
rejected with a message that names the offending key and lists the allowed
set, so a misspelled key errors instead of running with a wrong default.
The internal functions then trust the loader.

The same key set governs the Julia and Python implementations, and the
committed instances live once, under problems/ at the repository root, shared by the two
trees; the structural checker under tools/ confirms that equality.
"""

from __future__ import annotations

import pathlib
import tomllib
from dataclasses import dataclass

from .models.firstpassage import BiasedWalk
from .models.manna import MannaSandpile

ALLOWED_KINDS = {
    "barrier_survival",
    "barrier_mfpt",
    "manna_avalanche",
    "manna_survival",
}

ALLOWED_FIELDS = {
    "name", "kind", "output",
    # BiasedWalk fields.
    "nbits", "rate_up", "rate_down", "start",
    # MannaSandpile fields.
    "sites", "grains", "nmax", "seed_site",
    # Observable and engine controls.
    "tmax", "tpoints", "smax", "nshots", "seed", "maxdim", "cutoff",
    "engines",
}

ALLOWED_ENGINES = {"exact", "kmc", "tt"}

DEFAULT_OUTPUT_DIR = "results"
DEFAULT_TMAX = 10.0
DEFAULT_TPOINTS = 21
DEFAULT_SMAX = 60
DEFAULT_NSHOTS = 2000
DEFAULT_SEED = 1
DEFAULT_MAXDIM = 64
DEFAULT_CUTOFF = 1e-10


@dataclass(frozen=True)
class ProblemInstance:
    """One validated run: a model, an observable kind, and engine controls."""

    name: str
    kind: str
    output: str
    nbits: int
    rate_up: float
    rate_down: float
    start: int
    sites: int
    grains: int
    nmax: int
    seed_site: int
    tmax: float
    tpoints: int
    smax: int
    nshots: int
    seed: int
    maxdim: int
    cutoff: float
    engines: tuple[str, ...]

    def model(self):
        """Instantiate the model the kind names."""
        if self.kind.startswith("barrier"):
            return BiasedWalk(self.nbits, self.rate_up, self.rate_down, self.start)
        return MannaSandpile(self.sites, self.grains, nmax=self.nmax,
                             seed_site=self.seed_site)


def load_problem(path: str | pathlib.Path) -> ProblemInstance:
    """Load and validate one TOML problem instance.

    Raises
    ------
    ValueError
        On an unrecognised field, an unknown kind or engine name, or a
        parameter outside its documented range. The message names the
        offending key and lists the allowed set.
    """
    path = pathlib.Path(path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(data) - ALLOWED_FIELDS)
    if unknown:
        raise ValueError(
            f"unrecognised problem field(s): {', '.join(unknown)}. "
            f"allowed fields: {', '.join(sorted(ALLOWED_FIELDS))}"
        )
    kind = str(data.get("kind", "barrier_survival"))
    if kind not in ALLOWED_KINDS:
        raise ValueError(
            f"unknown kind: {kind}. allowed kinds: {', '.join(sorted(ALLOWED_KINDS))}"
        )
    engines = data.get("engines", sorted(ALLOWED_ENGINES))
    if isinstance(engines, str):
        engines = [engines]
    bad = sorted(set(engines) - ALLOWED_ENGINES)
    if bad:
        raise ValueError(
            f"unknown engine(s): {', '.join(bad)}. "
            f"allowed engines: {', '.join(sorted(ALLOWED_ENGINES))}"
        )
    nbits = int(data.get("nbits", 4))
    sites = int(data.get("sites", 6))
    grains = int(data.get("grains", 4))
    instance = ProblemInstance(
        name=str(data.get("name", path.stem)),
        kind=kind,
        output=str(data.get("output", DEFAULT_OUTPUT_DIR)),
        nbits=nbits,
        rate_up=float(data.get("rate_up", 1.0)),
        rate_down=float(data.get("rate_down", 1.0)),
        start=int(data.get("start", 2 ** (nbits - 1))),
        sites=sites,
        grains=grains,
        nmax=int(data.get("nmax", grains)),
        seed_site=int(data.get("seed_site", sites // 2)),
        tmax=float(data.get("tmax", DEFAULT_TMAX)),
        tpoints=int(data.get("tpoints", DEFAULT_TPOINTS)),
        smax=int(data.get("smax", DEFAULT_SMAX)),
        nshots=int(data.get("nshots", DEFAULT_NSHOTS)),
        seed=int(data.get("seed", DEFAULT_SEED)),
        maxdim=int(data.get("maxdim", DEFAULT_MAXDIM)),
        cutoff=float(data.get("cutoff", DEFAULT_CUTOFF)),
        engines=tuple(engines),
    )
    if instance.tmax <= 0:
        raise ValueError("tmax must be positive")
    if instance.tpoints < 2:
        raise ValueError("tpoints must be at least 2")
    if instance.smax < 1:
        raise ValueError("smax must be at least 1")
    if instance.nshots < 2:
        raise ValueError("nshots must be at least 2")
    if instance.maxdim < 1:
        raise ValueError("maxdim must be at least 1")
    if not 0 <= instance.cutoff < 1:
        raise ValueError("cutoff must lie in [0, 1)")
    instance.model()
    return instance


def problem_paths(names: list[str], directory: str | pathlib.Path) -> list[pathlib.Path]:
    """Resolve named instances under problems/; the name all expands to every file."""
    directory = pathlib.Path(directory)
    paths: list[pathlib.Path] = []
    for name in names:
        if name == "all":
            paths.extend(sorted(directory.glob("*.toml")))
        else:
            p = pathlib.Path(name) if name.endswith(".toml") else directory / f"{name}.toml"
            if not p.is_file():
                raise ValueError(f"problem instance not found: {p}")
            paths.append(p)
    return paths
