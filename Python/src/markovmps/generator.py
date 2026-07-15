"""The Markov generator: one assembly, written once.

A continuous-time Markov process is defined by its off-diagonal rates: the
entry W[y, x] is the rate of the move from configuration x to
configuration y. Probability conservation then fixes the diagonal: the
escape rate out of x is the sum of the column above it, so every column of
the full generator sums to 0 and dp/dt = W p conserves total probability.
The escape diagonal is the one place a sign error corrupts every result,
so it is assembled here once, from the off-diagonal part itself, and every
model reuses this function rather than restating the bookkeeping.

The assembly takes a model's terms: each term is a rate together with
either a mapping from sites to local matrices, which becomes a rank-1
product operator, or a prebuilt tensor-train operator for moves that are
not site-local products, such as the carry-chain shifts of the biased
walk. The column sums are computed by applying the transpose of the
off-diagonal operator to the all-ones vector, so a move blocked at a
boundary or an occupation cap is excluded from the escape rate
automatically, with no per-model boundary bookkeeping.
"""

from __future__ import annotations

import numpy as np

from .tensor_train import DEFAULT_CUTOFF, OPERATOR_MAXDIM, TT, TTOperator, mpo_sum

TermSpec = tuple[float, "dict[int, np.ndarray] | TTOperator"]


def offdiagonal_operator(dims: list[int], terms: list[TermSpec],
                         cutoff: float = DEFAULT_CUTOFF) -> TTOperator:
    """Sum the model's transition terms into one off-diagonal operator.

    Parameters
    ----------
    dims : list of int
        Local dimension at each site.
    terms : list of (rate, spec)
        spec is either a dict mapping site index to a local matrix, turned
        into a rank-1 product operator, or an already built TTOperator.
    cutoff : float
        Truncation threshold applied after each addition; the rank is
        uncapped because this object is part of the system, not the
        solution.

    Returns
    -------
    TTOperator
        The off-diagonal part of the generator, columns of absorbing
        configurations equal to 0 because no term moves out of them.

    Raises
    ------
    ValueError
        If the term list is empty or a rate is not positive.
    """
    if not terms:
        raise ValueError("a model must provide at least one transition term")
    operators: list[TTOperator] = []
    for rate, spec in terms:
        if rate <= 0:
            raise ValueError(f"transition rates must be positive, got {rate}")
        if isinstance(spec, TTOperator):
            operators.append(spec.scale(rate))
        else:
            operators.append(TTOperator.product_operator(dims, spec).scale(rate))
    return mpo_sum(operators, cutoff=cutoff, maxdim=OPERATOR_MAXDIM)


def escape_rates(offdiag: TTOperator, cutoff: float = DEFAULT_CUTOFF) -> TT:
    """Column sums of the off-diagonal operator: the escape rate at each state.

    Computed as the transpose applied to the all-ones vector, so a blocked
    move contributes nothing, matching the model's moves() exactly.
    """
    ones = TT.ones(offdiag.dims_out)
    return offdiag.transpose().apply(ones, cutoff=cutoff, maxdim=OPERATOR_MAXDIM)


def markov_generator(dims: list[int], terms: list[TermSpec],
                     cutoff: float = DEFAULT_CUTOFF) -> tuple[TTOperator, TTOperator, TT]:
    """Assemble the full generator W = T - diag(escape) from a model's terms.

    Returns
    -------
    (generator, offdiagonal, escape) : (TTOperator, TTOperator, TT)
        The full generator with columns summing to 0, the off-diagonal
        part T, and the escape-rate vector, returned separately because
        the avalanche generating function tilts T while keeping the
        escape diagonal fixed.
    """
    offdiag = offdiagonal_operator(dims, terms, cutoff=cutoff)
    escape = escape_rates(offdiag, cutoff=cutoff)
    diagonal = TTOperator.diagonal_from_tt(escape)
    generator = offdiag.add(diagonal.scale(-1.0)).compress(cutoff=cutoff, maxdim=OPERATOR_MAXDIM)
    return generator, offdiag, escape
