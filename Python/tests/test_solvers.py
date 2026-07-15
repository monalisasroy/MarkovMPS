"""Solver verbs against dense references, and the two baselines against each other."""

import numpy as np
import pytest
from scipy.linalg import expm

from markovmps import BiasedWalk, MannaSandpile, TT, max_abs_deviation
from markovmps.models.firstpassage import walk_generator
from markovmps.solvers import (
    exact_avalanche_size_distribution,
    exact_generator,
    exact_mfpt,
    exact_survival,
    kmc_avalanche_sizes,
    kmc_mfpt,
    kmc_survival,
    tt_evolve,
    tt_linear_solve,
)


def test_tt_evolve_matches_dense_exponential():
    walk = BiasedWalk(3, 1.1, 0.9, start=3)
    generator, _, _ = walk_generator(walk)
    dense = generator.to_dense()
    p0 = walk.initial_tt()
    t = 1.5
    reference = expm(dense * t) @ p0.to_dense()
    coarse = tt_evolve(generator, p0, t, dt=0.1, order=4).to_dense()
    fine = tt_evolve(generator, p0, t, dt=0.05, order=4).to_dense()
    # Measured 1.3e-7 at dt = 0.05 on this case; bounded at 1e-6 to hold
    # across linear-algebra backends.
    assert max_abs_deviation(fine, reference) < 1e-6
    # Halving the step must cut the global error by close to 2**4; the
    # measured ratio is 18, bounded below at 10 so the order is pinned.
    ratio = max_abs_deviation(coarse, reference) / max_abs_deviation(fine, reference)
    assert ratio > 10


def test_tt_evolve_conserves_probability():
    pile = MannaSandpile(4, 3)
    from markovmps.models.manna import sandpile_generator
    generator, _, _ = sandpile_generator(pile)
    p = tt_evolve(generator, pile.initial_tt(), 1.0, dt=0.05, order=4)
    assert abs(p.sum_entries().real - 1.0) < 1e-9


def test_tt_evolve_rejects_bad_steps():
    walk = BiasedWalk(3, 1.0, 1.0, start=3)
    generator, _, _ = walk_generator(walk)
    with pytest.raises(ValueError):
        tt_evolve(generator, walk.initial_tt(), -1.0, dt=0.1)
    with pytest.raises(ValueError):
        tt_evolve(generator, walk.initial_tt(), 1.0, dt=0.0)
    with pytest.raises(ValueError):
        tt_evolve(generator, walk.initial_tt(), 1.0, dt=0.1, order=0)


def test_tt_linear_solve_against_dense():
    # A small non-symmetric but well-conditioned system solved both ways.
    walk = BiasedWalk(3, 1.2, 0.8, start=3)
    generator, _, _ = walk_generator(walk)
    absorb = walk.absorbing_projector_tt()
    A = absorb.add(generator.transpose().scale(-1.0)).compress()
    ones = TT.ones(walk.local_dims())
    b = ones.add(absorb.apply(ones).scale(-1.0)).compress()
    x, residual = tt_linear_solve(A, b, sweeps=6)
    reference = np.linalg.solve(A.to_dense(), b.to_dense())
    assert residual < 1e-10
    assert max_abs_deviation(x.to_dense(), reference) < 1e-8


def test_exact_and_kmc_agree_on_the_walk():
    walk = BiasedWalk(3, 1.2, 0.8, start=3)
    mean, err = kmc_mfpt(walk, nshots=3000, seed=7)
    # 5 standard errors bounds the statistical fluctuation of the sampler.
    assert abs(mean - exact_mfpt(walk)) < 5 * err
    times = np.array([1.0, 3.0])
    kmc = kmc_survival(walk, times, nshots=2000, seed=5)
    exact = exact_survival(walk, times)
    assert max_abs_deviation(kmc, exact) < 0.05


def test_exact_and_kmc_agree_on_the_sandpile():
    pile = MannaSandpile(5, 4)
    sizes = kmc_avalanche_sizes(pile, nshots=3000, seed=11)
    dist = exact_avalanche_size_distribution(pile, 2000)
    mean_exact = float(np.sum(np.arange(2001) * dist))
    err = sizes.std(ddof=1) / np.sqrt(len(sizes))
    assert abs(sizes.mean() - mean_exact) < 5 * err


def test_exact_avalanche_distribution_normalises():
    pile = MannaSandpile(4, 3)
    dist = exact_avalanche_size_distribution(pile, 800)
    assert dist.min() > -1e-14
    assert abs(dist.sum() - 1.0) < 1e-10


def test_dense_state_limit_guard():
    pile = MannaSandpile(3, 2)
    W, states, _ = exact_generator(pile)
    assert W.shape == (len(states), len(states))
    # The counter and the enumeration agree where both are affordable,
    # which is what licenses guarding on the counter alone.
    assert pile.state_count() == len(states)
    walk = BiasedWalk(3, 1.0, 1.0, 3)
    assert walk.state_count() == len(walk.enumerate_states())
    # The guard runs before the enumeration, so refusing this model is
    # instant; enumerating its conserved sector (about 1e12 states) was
    # the allocation that once killed the process.
    with pytest.raises(ValueError):
        exact_generator(MannaSandpile(24, 21))
    import markovmps.solvers as solvers
    original = solvers.DENSE_STATE_LIMIT
    solvers.DENSE_STATE_LIMIT = 2
    try:
        with pytest.raises(ValueError):
            exact_generator(pile)
    finally:
        solvers.DENSE_STATE_LIMIT = original
