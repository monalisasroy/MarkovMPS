"""Problem loader: strict-key validation and the committed instances."""

import pathlib

import pytest

from markovmps import load_problem, problem_paths
from markovmps.models.firstpassage import BiasedWalk
from markovmps.models.manna import MannaSandpile

PROBLEMS = pathlib.Path(__file__).resolve().parents[2] / "problems"


def test_committed_instances_load_and_build():
    for path in sorted(PROBLEMS.glob("*.toml")):
        instance = load_problem(path)
        model = instance.model()
        assert isinstance(model, (BiasedWalk, MannaSandpile))


def test_unknown_field_is_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "bad"\nkind = "barrier_survival"\nnbitss = 4\n')
    with pytest.raises(ValueError, match="nbitss"):
        load_problem(bad)


def test_unknown_kind_is_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "bad"\nkind = "barrier_speed"\n')
    with pytest.raises(ValueError, match="barrier_speed"):
        load_problem(bad)


def test_unknown_engine_is_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('kind = "barrier_mfpt"\nengines = ["exact", "dmrg"]\n')
    with pytest.raises(ValueError, match="dmrg"):
        load_problem(bad)


def test_out_of_range_values_are_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('kind = "barrier_survival"\ntmax = -1.0\n')
    with pytest.raises(ValueError, match="tmax"):
        load_problem(bad)


def test_problem_paths_all_and_missing(tmp_path):
    everything = problem_paths(["all"], PROBLEMS)
    assert len(everything) >= 4
    with pytest.raises(ValueError):
        problem_paths(["nonexistent"], PROBLEMS)
