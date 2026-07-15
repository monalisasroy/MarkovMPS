"""Generator assembly: the twin check and the column-sum-zero invariant."""

import itertools

import numpy as np
import pytest

from markovmps import BiasedWalk, MannaSandpile, markov_generator
from markovmps.generator import offdiagonal_operator
from markovmps.solvers import exact_generator
from markovmps.tensor_train import TT


@pytest.mark.parametrize("nbits,up,down", [(3, 1.0, 1.0), (4, 1.3, 0.7), (5, 0.4, 1.9)])
def test_walk_generator_twin_and_column_sums(nbits, up, down):
    walk = BiasedWalk(nbits, up, down, start=1)
    W_tt, _, _ = markov_generator(walk.local_dims(), walk.terms())
    dense_tt = W_tt.to_dense()
    dense_ref, _, _ = exact_generator(walk)
    # The tensor-train assembly and the moves() enumeration are independent
    # constructions of the same generator; their agreement checks both.
    assert np.max(np.abs(dense_tt - dense_ref)) < 1e-11
    assert np.max(np.abs(dense_tt.sum(axis=0))) < 1e-11


@pytest.mark.parametrize("nsites,grains", [(3, 2), (4, 3)])
def test_manna_generator_twin_and_column_sums(nsites, grains):
    pile = MannaSandpile(nsites, grains)
    W_tt, _, escape = markov_generator(pile.local_dims(), pile.terms())
    dense_tt = W_tt.to_dense()
    dims = pile.local_dims()
    states = list(itertools.product(*[range(d) for d in dims]))
    index = {s: i for i, s in enumerate(states)}
    dense_ref = np.zeros((len(states), len(states)))
    for s in states:
        i = index[s]
        for rate, dest in pile.moves(s):
            dense_ref[index[dest], i] += rate
            dense_ref[i, i] -= rate
    assert np.max(np.abs(dense_tt - dense_ref)) < 1e-10
    assert np.max(np.abs(dense_tt.sum(axis=0))) < 1e-10
    # On the conserved sector the escape rate equals the active-site count.
    activity = pile.activity_tt().to_dense()
    escape_dense = escape.to_dense()
    sector = [index[s] for s in pile.enumerate_states()]
    assert np.max(np.abs(activity[sector] - escape_dense[sector])) < 1e-10


def test_ones_annihilated_by_generator():
    # 1^T W = 0 is probability conservation, checked in tensor-train form.
    pile = MannaSandpile(4, 3)
    W_tt, _, _ = markov_generator(pile.local_dims(), pile.terms())
    ones = TT.ones(pile.local_dims())
    assert W_tt.transpose().apply(ones).norm() < 1e-9


def test_rejects_bad_terms():
    with pytest.raises(ValueError):
        offdiagonal_operator([2, 2], [])
    with pytest.raises(ValueError):
        offdiagonal_operator([2, 2], [(-1.0, {0: np.eye(2)})])
