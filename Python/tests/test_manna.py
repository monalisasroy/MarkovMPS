"""MannaSandpile: sector structure, and the tensor-train solves against the dense twin."""

import numpy as np
import pytest

from markovmps import MannaSandpile
from markovmps.models import manna as mn
from markovmps.solvers import (
    exact_avalanche_pgf,
    exact_avalanche_size_distribution,
    exact_survival,
)


def test_seeded_configuration_is_deterministic():
    pile = MannaSandpile(6, 4)
    occ = pile.initial_state()
    assert sum(occ) == 4
    assert occ[pile.seed_site] == 2
    assert pile.activity(occ) == 1
    assert MannaSandpile(6, 4).initial_state() == occ


def test_moves_conserve_grains():
    pile = MannaSandpile(5, 4)
    for state in pile.enumerate_states():
        for rate, dest in pile.moves(state):
            assert sum(dest) == pile.grains
            assert rate > 0


def test_sector_enumeration_counts():
    # Compositions of 3 grains over 4 sites capped at 3: C(3+4-1, 3) = 20.
    pile = MannaSandpile(4, 3)
    assert len(pile.enumerate_states()) == 20


def test_grain_sector_projector_matches_enumeration():
    import itertools
    pile = MannaSandpile(4, 3)
    projector = pile.grain_sector_projector_tt().to_dense()
    states = list(itertools.product(*[range(d) for d in pile.local_dims()]))
    reference = np.diag([1.0 if sum(s) == pile.grains else 0.0 for s in states])
    assert np.allclose(projector, reference)


def test_tt_survival_against_exact():
    pile = MannaSandpile(4, 3)
    times = np.array([0.5, 1.5, 3.0])
    observed = mn.tt_survival(pile, times, maxdim=48)
    reference = exact_survival(pile, times)
    # Bound: truncated-Taylor step error at dt = 0.05; measured 1.2e-9 at
    # N = 5, bounded at 1e-6 to hold across backends.
    assert np.max(np.abs(observed - reference)) < 1e-6


@pytest.mark.parametrize("z", [0.5, 0.9, np.exp(0.4j)])
def test_tt_pgf_against_dense_twin(z):
    pile = MannaSandpile(4, 3)
    value, residual, _ = mn.tt_avalanche_pgf(pile, z, maxdim=48)
    assert residual < 1e-10
    assert abs(value - exact_avalanche_pgf(pile, z)) < 1e-9


def test_tt_mean_avalanche_size_against_exact():
    pile = MannaSandpile(4, 3)
    mean_tt, residual = mn.tt_mean_avalanche_size(pile, maxdim=48)
    assert residual < 1e-10
    dist = exact_avalanche_size_distribution(pile, 800)
    mean_exact = float(np.sum(np.arange(801) * dist))
    assert abs(mean_tt - mean_exact) < 1e-8


def test_tt_size_distribution_against_exact():
    pile = MannaSandpile(4, 3)
    smax = 20
    observed = mn.tt_avalanche_size_distribution(pile, smax=smax, maxdim=48, sweeps=6)
    reference = exact_avalanche_size_distribution(pile, smax)
    tail = 1.0 - exact_avalanche_size_distribution(pile, 2 * smax + 1).sum()
    # The Fourier inversion aliases the tail beyond 2 smax + 2 points, so
    # the deviation is bounded by that tail mass plus solver tolerance.
    assert np.max(np.abs(observed - reference)) < tail + 1e-7


def test_model_validation():
    with pytest.raises(ValueError):
        MannaSandpile(2, 2)          # ring too short
    with pytest.raises(ValueError):
        MannaSandpile(4, 1)          # no active seed possible
    with pytest.raises(ValueError):
        MannaSandpile(4, 5)          # no absorbing configuration
    with pytest.raises(ValueError):
        MannaSandpile(4, 3, nmax=1)  # no active site representable
    with pytest.raises(ValueError):
        MannaSandpile(4, 3, seed_site=9)


def test_solution_cap_does_not_corrupt_the_operator():
    """Regression: a solution rank cap below the operator's natural rank.

    The assembly once compressed the operator itself under the solution's
    maxdim; at maxdim 8, below the middle-bond rank 15 of this operator,
    the solve then satisfied a corrupted generator to machine precision
    while the mean was wrong by half its value. Operators are now
    compressed by the cutoff alone, so a small solution cap must either
    reproduce the exact mean or report a residual on the scale of its
    error, never a tiny residual beside a wrong answer.
    """
    pile = MannaSandpile(4, 3)
    dist = exact_avalanche_size_distribution(pile, 800)
    mean_exact = float(np.sum(np.arange(801) * dist))
    for maxdim in (4, 8):
        mean_tt, residual = mn.tt_mean_avalanche_size(pile, maxdim=maxdim)
        error = abs(mean_tt - mean_exact)
        assert not (residual < 1e-8 and error > 1e-6), (
            f"maxdim {maxdim}: residual {residual:.1e} certifies a wrong mean, error {error:.1e}")
    mean_tt, _ = mn.tt_mean_avalanche_size(pile, maxdim=8)
    assert abs(mean_tt - mean_exact) < 1e-9
