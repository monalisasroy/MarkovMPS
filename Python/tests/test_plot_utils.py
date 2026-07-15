"""Writer-and-plotting layer: every kind writes both files, and tags never collide."""

import pathlib

import pytest

from markovmps.plot_utils import flag_tag, run_problem
from markovmps.utils import load_problem

PROBLEMS = pathlib.Path(__file__).resolve().parents[2] / "problems"


@pytest.mark.parametrize("name", ["barrier", "barrier_mfpt", "manna_small"])
def test_every_kind_writes_table_and_figure(name, tmp_path):
    instance = load_problem(PROBLEMS / f"{name}.toml")
    # Shrink the sampling so the test stays fast; the committed instance
    # keeps its own full parameters for the reproducible run.
    small = type(instance)(**{**instance.__dict__, "nshots": 50, "tpoints": 5,
                              "smax": min(instance.smax, 12)})
    destination = run_problem(small, outdir=tmp_path)
    assert destination.table.is_file() and destination.table.stat().st_size > 0
    assert destination.figure.is_file() and destination.figure.stat().st_size > 0
    assert destination.figure.parent == tmp_path / "figures"


def test_flag_tags_never_collide():
    tags = [flag_tag("manna_avalanche", 4, 1.0, 1.0, 8, sites, 4, 10.0, 21, 60, 2000, 1)
            for sites in (4, 5, 6)]
    assert len(tags) == len(set(tags))
    # A kind that reads only some fields must still change tag when any of
    # the fields it reads changes, or a sweep silently overwrites output.
    tags_seed = [flag_tag("barrier_mfpt", 4, 1.0, 1.0, 8, 6, 4, 10.0, 21, 60, 2000, seed)
                 for seed in (1, 2, 3)]
    assert len(tags_seed) == len(set(tags_seed))


def test_flag_tag_rejects_unknown_kind():
    with pytest.raises(ValueError):
        flag_tag("barrier_speed", 4, 1.0, 1.0, 8, 6, 4, 10.0, 21, 60, 2000, 1)
