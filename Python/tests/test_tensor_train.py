"""Tensor-train core against dense linear algebra."""

import numpy as np
import pytest

from markovmps.tensor_train import TT, TTOperator, mpo_sum, qtt_shift

RNG = np.random.default_rng(0)
DIMS = [2, 3, 2, 4]


def test_dense_round_trip():
    vec = RNG.normal(size=int(np.prod(DIMS)))
    assert np.allclose(TT.from_dense(vec, DIMS).to_dense(), vec)


def test_product_state_ordering():
    values = [1, 2, 0, 3]
    flat = ((values[0] * 3 + values[1]) * 2 + values[2]) * 4 + values[3]
    dense = TT.product_state(DIMS, values).to_dense()
    assert dense[flat] == 1.0
    assert dense.sum() == 1.0


def test_add_scale_inner_sum():
    v = RNG.normal(size=int(np.prod(DIMS)))
    w = RNG.normal(size=v.size)
    tv, tw = TT.from_dense(v, DIMS), TT.from_dense(w, DIMS)
    assert np.allclose(tv.add(tw.scale(-2.0)).compress().to_dense(), v - 2 * w)
    assert np.isclose(tv.inner(tw), np.dot(v, w))
    assert np.isclose(tv.sum_entries(), v.sum())


def test_compress_reduces_rank_and_preserves_values():
    v = RNG.normal(size=int(np.prod(DIMS)))
    w = RNG.normal(size=v.size)
    fat = TT.from_dense(v, DIMS).add(TT.from_dense(w, DIMS))
    slim = fat.compress()
    assert max(slim.ranks) <= max(fat.ranks)
    assert np.allclose(slim.to_dense(), v + w)


def test_operator_apply_transpose_compose():
    m1 = RNG.normal(size=(3, 3))
    m2 = RNG.normal(size=(4, 4))
    op = TTOperator.product_operator(DIMS, {1: m1, 3: m2})
    dense = np.kron(np.kron(np.kron(np.eye(2), m1), np.eye(2)), m2)
    v = RNG.normal(size=int(np.prod(DIMS)))
    tv = TT.from_dense(v, DIMS)
    assert np.allclose(op.to_dense(), dense)
    assert np.allclose(op.apply(tv, cutoff=0.0).to_dense(), dense @ v)
    assert np.allclose(op.transpose().to_dense(), dense.T)
    assert np.allclose(op.compose(op).to_dense(), dense @ dense)


def test_mpo_sum_and_diagonal():
    m0 = RNG.normal(size=(2, 2))
    op0 = TTOperator.product_operator(DIMS, {0: m0})
    op1 = TTOperator.identity(DIMS)
    total = mpo_sum([op0, op1.scale(0.5)])
    dense0 = np.kron(np.kron(np.kron(m0, np.eye(3)), np.eye(2)), np.eye(4))
    assert np.allclose(total.to_dense(), dense0 + 0.5 * np.eye(int(np.prod(DIMS))))
    diag_vec = RNG.normal(size=int(np.prod(DIMS)))
    diag_op = TTOperator.diagonal_from_tt(TT.from_dense(diag_vec, DIMS))
    assert np.allclose(diag_op.to_dense(), np.diag(diag_vec))


@pytest.mark.parametrize("nbits", range(1, 7))
def test_qtt_shift_matches_dense(nbits):
    n = 2 ** nbits
    assert np.allclose(qtt_shift(nbits, +1).to_dense(), np.eye(n, k=-1))
    assert np.allclose(qtt_shift(nbits, -1).to_dense(), np.eye(n, k=+1))


def test_qtt_shift_rank_is_2():
    ranks = qtt_shift(6, +1).ranks
    assert ranks == [1] + [2] * 5 + [1]


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        TT([np.zeros((2, 2, 1))])       # boundary rank not 1
    with pytest.raises(ValueError):
        TT.product_state([2, 2], [0, 5])  # value out of range
    with pytest.raises(ValueError):
        qtt_shift(3, 2)                  # step must be +1 or -1
