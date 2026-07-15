"""BiasedWalk: closed forms against the dense engine, tensor train against both."""

import numpy as np
import pytest

from markovmps import BiasedWalk, exact_mfpt, exact_survival
from markovmps.models import firstpassage as fp
from markovmps.solvers import exact_absorption_probability, exact_generator

CASES = [(3, 1.0, 1.0, 3), (4, 1.3, 0.7, 5), (4, 0.6, 1.5, 9)]


@pytest.mark.parametrize("nbits,up,down,start", CASES)
def test_closed_forms_against_dense(nbits, up, down, start):
    walk = BiasedWalk(nbits, up, down, start)
    split_dense = exact_absorption_probability(walk, lambda s: s[0] == walk.last)
    assert abs(split_dense - walk.analytic_splitting()) < 1e-10
    assert abs(exact_mfpt(walk) - walk.analytic_mfpt()) < 1e-9
    times = np.linspace(0.0, 6.0, 7)
    assert np.max(np.abs(exact_survival(walk, times) - walk.analytic_survival(times))) < 1e-10


@pytest.mark.parametrize("nbits,up,down,start", CASES)
def test_spectrum_closed_form(nbits, up, down, start):
    walk = BiasedWalk(nbits, up, down, start)
    W, states, _ = exact_generator(walk)
    transient = [i for i, s in enumerate(states) if not walk.is_absorbing(s)]
    Q = W[np.ix_(transient, transient)]
    eigen = np.sort(np.linalg.eigvals(Q).real)
    assert np.max(np.abs(eigen - np.sort(-walk.analytic_decay_rates()))) < 1e-10


def test_tt_survival_against_closed_form():
    walk = BiasedWalk(4, 1.3, 0.7, start=5)
    times = np.array([0.5, 1.0, 2.0, 4.0])
    observed = fp.tt_survival(walk, times, maxdim=64)
    # Bound: the truncated-Taylor global step error at dt = 0.1; measured
    # 1.6e-8 on this case, bounded at 1e-6 to hold across backends.
    assert np.max(np.abs(observed - walk.analytic_survival(times))) < 1e-6


def test_tt_mfpt_and_splitting_against_closed_forms():
    walk = BiasedWalk(4, 1.3, 0.7, start=5)
    mfpt, residual = fp.tt_mfpt(walk)
    assert residual < 1e-10
    assert abs(mfpt - walk.analytic_mfpt()) < 1e-8
    split, residual = fp.tt_splitting(walk)
    assert residual < 1e-10
    assert abs(split - walk.analytic_splitting()) < 1e-8


def test_tt_slowest_decay_rate_against_closed_form():
    walk = BiasedWalk(4, 1.3, 0.7, start=5)
    observed = fp.tt_slowest_decay_rate(walk)
    expected = walk.analytic_decay_rates()[0]
    assert abs(observed - expected) < 1e-9


def test_unbiased_limits():
    walk = BiasedWalk(4, 1.0, 1.0, start=5)
    length = walk.last
    assert abs(walk.analytic_splitting() - 5 / length) < 1e-14
    assert abs(walk.analytic_mfpt() - 5 * (length - 5) / 2.0) < 1e-12


def test_model_validation():
    with pytest.raises(ValueError):
        BiasedWalk(1, 1.0, 1.0, 1)
    with pytest.raises(ValueError):
        BiasedWalk(3, -1.0, 1.0, 3)
    with pytest.raises(ValueError):
        BiasedWalk(3, 1.0, 1.0, 0)   # absorbing start
    with pytest.raises(ValueError):
        BiasedWalk(3, 1.0, 1.0, 7)   # absorbing start
