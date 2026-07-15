"""Tensor-train vectors and operators for classical probability distributions.

A probability distribution over N sites with local dimension d has d**N
entries. The tensor train stores it as N cores of shape
(rank_left, d, rank_right), at a cost set by the correlations in the
distribution rather than by the size of the state space. This module holds
the dependency-free core: the vector class, the operator class, the
compression sweep, and the binary-address shift operators used by the
first-passage model. Nothing in this file knows about Markov generators;
that assembly lives in generator.py.

Conventions, stated once for the whole package:

- A TT vector core has shape (rank_left, d, rank_right); the first core has
  rank_left = 1 and the last has rank_right = 1.
- A TT operator core has shape (rank_left, d_out, d_in, rank_right).
- Operator entries follow the column convention of the generator: entry
  W[y, x] is the rate from state x to state y, so a distribution evolves as
  dp/dt = W p and probability conservation reads: every column sums to 0.
- Compression is a left-to-right QR sweep followed by a right-to-left SVD
  sweep; singular values below cutoff times the largest are dropped, and at
  most maxdim are kept per bond.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CUTOFF = 1e-12
DEFAULT_MAXDIM = 256
# Rank caps apply to solution tensors only. An operator or right-hand side
# compressed under a rank cap is a different system, which a solver can then
# satisfy to machine precision while answering the wrong question, so system
# objects are compressed by the cutoff alone under this effectively
# unbounded cap.
OPERATOR_MAXDIM = 2**30


def _validate_cores(cores: list[np.ndarray], ndim: int, label: str) -> None:
    """Check core shapes and the rank chain of a tensor train.

    Parameters
    ----------
    cores : list of ndarray
        Candidate cores.
    ndim : int
        3 for a vector core, 4 for an operator core.
    label : str
        Name used in error messages.

    Raises
    ------
    ValueError
        If the list is empty, a core has the wrong number of axes, the
        boundary ranks are not 1, or adjacent ranks do not match.
    """
    if not cores:
        raise ValueError(f"{label} needs at least one core")
    for k, core in enumerate(cores):
        if core.ndim != ndim:
            raise ValueError(f"{label} core {k} has {core.ndim} axes, expected {ndim}")
    if cores[0].shape[0] != 1 or cores[-1].shape[-1] != 1:
        raise ValueError(f"{label} boundary ranks must be 1")
    for k in range(len(cores) - 1):
        if cores[k].shape[-1] != cores[k + 1].shape[0]:
            raise ValueError(
                f"{label} rank mismatch between cores {k} and {k + 1}: "
                f"{cores[k].shape[-1]} against {cores[k + 1].shape[0]}"
            )


class TT:
    """A tensor-train vector: a distribution or a general vector over N sites."""

    def __init__(self, cores: list[np.ndarray]):
        _validate_cores(cores, 3, "TT vector")
        self.cores = [np.asarray(c) for c in cores]

    @property
    def nsites(self) -> int:
        """Number of sites."""
        return len(self.cores)

    @property
    def dims(self) -> list[int]:
        """Local dimension at each site."""
        return [c.shape[1] for c in self.cores]

    @property
    def ranks(self) -> list[int]:
        """Bond ranks, length nsites + 1, boundaries included."""
        return [self.cores[0].shape[0]] + [c.shape[2] for c in self.cores]

    def copy(self) -> "TT":
        """Deep copy."""
        return TT([c.copy() for c in self.cores])

    @staticmethod
    def product_state(dims: list[int], values: list[int]) -> "TT":
        """Rank-1 delta distribution on one configuration.

        Parameters
        ----------
        dims : list of int
            Local dimension at each site.
        values : list of int
            The configuration; values[i] must lie in range(dims[i]).

        Returns
        -------
        TT
            The distribution that puts all mass on the given configuration.

        Raises
        ------
        ValueError
            If the lengths differ or a value is out of range.
        """
        if len(dims) != len(values):
            raise ValueError("dims and values must have the same length")
        cores = []
        for d, v in zip(dims, values):
            if not 0 <= v < d:
                raise ValueError(f"value {v} out of range for local dimension {d}")
            core = np.zeros((1, d, 1))
            core[0, v, 0] = 1.0
            cores.append(core)
        return TT(cores)

    @staticmethod
    def ones(dims: list[int]) -> "TT":
        """Rank-1 all-ones vector, used to sum a distribution's entries."""
        return TT([np.ones((1, d, 1)) for d in dims])

    @staticmethod
    def from_dense(vec: np.ndarray, dims: list[int], cutoff: float = 0.0,
                   maxdim: int = DEFAULT_MAXDIM) -> "TT":
        """Exact TT decomposition of a dense vector by sequential SVD.

        The first site is the slowest-varying index: entry vec[j] with
        j = values[0] * prod(dims[1:]) + ... matches product_state ordering.
        """
        total = int(np.prod(dims))
        if vec.size != total:
            raise ValueError(f"vector length {vec.size} does not match dims product {total}")
        cores = []
        rest = vec.reshape(1, total)
        rank = 1
        for k, d in enumerate(dims[:-1]):
            m = rest.reshape(rank * d, -1)
            u, s, vh = np.linalg.svd(m, full_matrices=False)
            keep = _keep_count(s, cutoff, maxdim)
            u, s, vh = u[:, :keep], s[:keep], vh[:keep]
            cores.append(u.reshape(rank, d, keep))
            rest = (s[:, None] * vh)
            rank = keep
        cores.append(rest.reshape(rank, dims[-1], 1))
        return TT(cores)

    def to_dense(self) -> np.ndarray:
        """Contract to a dense vector; only for small state spaces."""
        total = int(np.prod(self.dims))
        if total > 2_000_000:
            raise ValueError(f"state space of size {total} is too large to densify")
        out = self.cores[0].reshape(self.cores[0].shape[1], self.cores[0].shape[2])
        for core in self.cores[1:]:
            out = np.tensordot(out, core, axes=([out.ndim - 1], [0]))
        return out.reshape(total)

    def scale(self, factor: complex) -> "TT":
        """Return the vector multiplied by a scalar."""
        cores = [c.copy() for c in self.cores]
        cores[0] = cores[0] * factor
        return TT(cores)

    def add(self, other: "TT") -> "TT":
        """Direct-sum addition; compress afterwards to control the rank."""
        if self.dims != other.dims:
            raise ValueError("cannot add TT vectors with different local dimensions")
        n = self.nsites
        if n == 1:
            return TT([self.cores[0] + other.cores[0]])
        cores = []
        for k in range(n):
            a, b = self.cores[k], other.cores[k]
            d = a.shape[1]
            dtype = np.result_type(a, b)
            if k == 0:
                core = np.zeros((1, d, a.shape[2] + b.shape[2]), dtype=dtype)
                core[:, :, : a.shape[2]] = a
                core[:, :, a.shape[2]:] = b
            elif k == n - 1:
                core = np.zeros((a.shape[0] + b.shape[0], d, 1), dtype=dtype)
                core[: a.shape[0]] = a
                core[a.shape[0]:] = b
            else:
                core = np.zeros((a.shape[0] + b.shape[0], d, a.shape[2] + b.shape[2]), dtype=dtype)
                core[: a.shape[0], :, : a.shape[2]] = a
                core[a.shape[0]:, :, a.shape[2]:] = b
            cores.append(core)
        return TT(cores)

    def inner(self, other: "TT") -> complex:
        """Euclidean inner product sum(conj(self) * other)."""
        if self.dims != other.dims:
            raise ValueError("cannot contract TT vectors with different local dimensions")
        env = np.ones((1, 1))
        for a, b in zip(self.cores, other.cores):
            env = np.einsum("ab,aic,bid->cd", env, a.conj(), b, optimize=True)
        return complex(env[0, 0])

    def sum_entries(self) -> complex:
        """Sum of all entries: the total probability of a distribution."""
        return TT.ones(self.dims).inner(self)

    def norm(self) -> float:
        """Euclidean norm."""
        return float(np.sqrt(abs(self.inner(self))))

    def compress(self, cutoff: float = DEFAULT_CUTOFF, maxdim: int = DEFAULT_MAXDIM) -> "TT":
        """Return a compressed copy: QR sweep left to right, SVD sweep back.

        Singular values below cutoff times the largest at each bond are
        dropped, and at most maxdim survive per bond.
        """
        cores = [c.copy() for c in self.cores]
        n = len(cores)
        for k in range(n - 1):
            rl, d, rr = cores[k].shape
            q, r = np.linalg.qr(cores[k].reshape(rl * d, rr))
            cores[k] = q.reshape(rl, d, q.shape[1])
            cores[k + 1] = np.tensordot(r, cores[k + 1], axes=([1], [0]))
        for k in range(n - 1, 0, -1):
            rl, d, rr = cores[k].shape
            m = cores[k].reshape(rl, d * rr)
            u, s, vh = np.linalg.svd(m, full_matrices=False)
            keep = _keep_count(s, cutoff, maxdim)
            u, s, vh = u[:, :keep], s[:keep], vh[:keep]
            cores[k] = vh.reshape(keep, d, rr)
            cores[k - 1] = np.tensordot(cores[k - 1], u * s, axes=([2], [0]))
        return TT(cores)


def _keep_count(s: np.ndarray, cutoff: float, maxdim: int) -> int:
    """Number of singular values kept under a relative cutoff and a hard cap."""
    if maxdim < 1:
        raise ValueError("maxdim must be at least 1")
    if s.size == 0:
        return 1
    keep = int(np.sum(s > cutoff * s[0]))
    return max(1, min(keep, maxdim))


class TTOperator:
    """A tensor-train operator (matrix product operator) over N sites."""

    def __init__(self, cores: list[np.ndarray]):
        _validate_cores(cores, 4, "TT operator")
        self.cores = [np.asarray(c) for c in cores]

    @property
    def nsites(self) -> int:
        """Number of sites."""
        return len(self.cores)

    @property
    def dims_out(self) -> list[int]:
        """Output local dimension at each site."""
        return [c.shape[1] for c in self.cores]

    @property
    def dims_in(self) -> list[int]:
        """Input local dimension at each site."""
        return [c.shape[2] for c in self.cores]

    @property
    def ranks(self) -> list[int]:
        """Bond ranks, length nsites + 1, boundaries included."""
        return [self.cores[0].shape[0]] + [c.shape[3] for c in self.cores]

    def copy(self) -> "TTOperator":
        """Deep copy."""
        return TTOperator([c.copy() for c in self.cores])

    @staticmethod
    def identity(dims: list[int]) -> "TTOperator":
        """Rank-1 identity operator."""
        return TTOperator([np.eye(d).reshape(1, d, d, 1) for d in dims])

    @staticmethod
    def product_operator(dims: list[int], site_matrices: dict[int, np.ndarray]) -> "TTOperator":
        """Rank-1 operator: the given matrix at each named site, identity elsewhere.

        Parameters
        ----------
        dims : list of int
            Local dimension at each site.
        site_matrices : dict mapping site index to a (d, d) matrix
            Sites are 0-based. Every named site must be in range.
        """
        cores = []
        for i, d in enumerate(dims):
            m = site_matrices.get(i)
            if m is None:
                m = np.eye(d)
            else:
                m = np.asarray(m)
                if m.shape != (d, d):
                    raise ValueError(f"matrix at site {i} has shape {m.shape}, expected ({d}, {d})")
            cores.append(m.reshape(1, d, d, 1))
        for site in site_matrices:
            if not 0 <= site < len(dims):
                raise ValueError(f"site index {site} out of range for {len(dims)} sites")
        return TTOperator(cores)

    @staticmethod
    def diagonal_from_tt(vec: TT) -> "TTOperator":
        """Diagonal operator whose diagonal is the given TT vector."""
        cores = []
        for c in vec.cores:
            rl, d, rr = c.shape
            core = np.zeros((rl, d, d, rr), dtype=c.dtype)
            for i in range(d):
                core[:, i, i, :] = c[:, i, :]
            cores.append(core)
        return TTOperator(cores)

    def transpose(self) -> "TTOperator":
        """Operator transpose: swap the input and output legs of every core."""
        return TTOperator([np.transpose(c, (0, 2, 1, 3)) for c in self.cores])

    def scale(self, factor: complex) -> "TTOperator":
        """Return the operator multiplied by a scalar."""
        cores = [c.copy() for c in self.cores]
        cores[0] = cores[0] * factor
        return TTOperator(cores)

    def add(self, other: "TTOperator") -> "TTOperator":
        """Direct-sum addition; compress afterwards to control the rank."""
        if self.dims_in != other.dims_in or self.dims_out != other.dims_out:
            raise ValueError("cannot add TT operators with different local dimensions")
        n = self.nsites
        if n == 1:
            return TTOperator([self.cores[0] + other.cores[0]])
        cores = []
        for k in range(n):
            a, b = self.cores[k], other.cores[k]
            do, di = a.shape[1], a.shape[2]
            dtype = np.result_type(a, b)
            if k == 0:
                core = np.zeros((1, do, di, a.shape[3] + b.shape[3]), dtype=dtype)
                core[:, :, :, : a.shape[3]] = a
                core[:, :, :, a.shape[3]:] = b
            elif k == n - 1:
                core = np.zeros((a.shape[0] + b.shape[0], do, di, 1), dtype=dtype)
                core[: a.shape[0]] = a
                core[a.shape[0]:] = b
            else:
                core = np.zeros(
                    (a.shape[0] + b.shape[0], do, di, a.shape[3] + b.shape[3]), dtype=dtype
                )
                core[: a.shape[0], :, :, : a.shape[3]] = a
                core[a.shape[0]:, :, :, a.shape[3]:] = b
            cores.append(core)
        return TTOperator(cores)

    def apply(self, vec: TT, cutoff: float = DEFAULT_CUTOFF,
              maxdim: int = DEFAULT_MAXDIM) -> TT:
        """Apply the operator to a TT vector and compress the result."""
        if self.dims_in != vec.dims:
            raise ValueError("operator input dimensions do not match the vector")
        cores = []
        for oc, vc in zip(self.cores, vec.cores):
            merged = np.einsum("aijb,cjd->acibd", oc, vc, optimize=True)
            rl = oc.shape[0] * vc.shape[0]
            rr = oc.shape[3] * vc.shape[2]
            cores.append(merged.reshape(rl, oc.shape[1], rr))
        return TT(cores).compress(cutoff=cutoff, maxdim=maxdim)

    def compose(self, other: "TTOperator", cutoff: float = DEFAULT_CUTOFF,
                maxdim: int = DEFAULT_MAXDIM) -> "TTOperator":
        """Operator product self other, contracted core by core and compressed."""
        if self.dims_in != other.dims_out:
            raise ValueError("operator dimensions do not match for composition")
        cores = []
        for ca, cb in zip(self.cores, other.cores):
            merged = np.einsum("aijb,cjkd->acikbd", ca, cb, optimize=True)
            rl = ca.shape[0] * cb.shape[0]
            rr = ca.shape[3] * cb.shape[3]
            cores.append(merged.reshape(rl, ca.shape[1], cb.shape[2], rr))
        return TTOperator(cores).compress(cutoff=cutoff, maxdim=maxdim)

    def compress(self, cutoff: float = DEFAULT_CUTOFF, maxdim: int = DEFAULT_MAXDIM) -> "TTOperator":
        """Return a compressed copy, reusing the vector compression sweep."""
        merged = []
        for c in self.cores:
            rl, do, di, rr = c.shape
            merged.append(c.reshape(rl, do * di, rr))
        compressed = TT(merged).compress(cutoff=cutoff, maxdim=maxdim)
        cores = []
        for c, orig in zip(compressed.cores, self.cores):
            rl, _, rr = c.shape
            cores.append(c.reshape(rl, orig.shape[1], orig.shape[2], rr))
        return TTOperator(cores)

    def to_dense(self) -> np.ndarray:
        """Contract to a dense matrix; only for small state spaces."""
        total_out = int(np.prod(self.dims_out))
        total_in = int(np.prod(self.dims_in))
        if total_out * total_in > 12_000_000:
            raise ValueError("operator too large to densify")
        env = self.cores[0]
        for core in self.cores[1:]:
            env = np.einsum("aijb,bklc->aikjlc", env, core, optimize=True)
            a, i, k, j, l, c = env.shape
            env = env.reshape(a, i * k, j * l, c)
        return env.reshape(total_out, total_in)


def tt_sum(vectors: list[TT], cutoff: float = DEFAULT_CUTOFF,
           maxdim: int = DEFAULT_MAXDIM) -> TT:
    """Sum a list of TT vectors, compressing after each addition."""
    if not vectors:
        raise ValueError("tt_sum needs at least one vector")
    out = vectors[0]
    for v in vectors[1:]:
        out = out.add(v).compress(cutoff=cutoff, maxdim=maxdim)
    return out


def mpo_sum(operators: list[TTOperator], cutoff: float = DEFAULT_CUTOFF,
            maxdim: int = DEFAULT_MAXDIM) -> TTOperator:
    """Sum a list of TT operators, compressing after each addition."""
    if not operators:
        raise ValueError("mpo_sum needs at least one operator")
    out = operators[0]
    for o in operators[1:]:
        out = out.add(o).compress(cutoff=cutoff, maxdim=maxdim)
    return out


def qtt_shift(nbits: int, step: int) -> TTOperator:
    """Binary-address shift operator on 2**nbits states, without wraparound.

    Entry [y, x] is 1 when y = x + step and both lie in range(2**nbits), for
    step in (+1, -1). The construction is the carry chain on the binary
    digits, with the first site the most significant bit, matching the
    ordering of TT.product_state and TT.from_dense; the shift by +1 flips
    trailing 1s to 0 and the first 0 above them to 1. The carry that leaves
    the most significant bit is dropped, which is what removes wraparound.
    Bond rank is 2 regardless of nbits.

    Parameters
    ----------
    nbits : int
        Number of binary sites; the state space has 2**nbits points.
    step : int
        +1 for the raising shift, -1 for the lowering shift.

    Returns
    -------
    TTOperator
        The shift operator with columns for out-of-range moves equal to zero.
    """
    if nbits < 1:
        raise ValueError("nbits must be at least 1")
    if step not in (1, -1):
        raise ValueError("step must be +1 or -1")
    raise_bit = np.zeros((2, 2))
    raise_bit[1, 0] = 1.0
    lower_bit = np.zeros((2, 2))
    lower_bit[0, 1] = 1.0
    eye = np.eye(2)
    if step == 1:
        carry, finish = lower_bit, raise_bit
    else:
        carry, finish = raise_bit, lower_bit
    # Bond states, ordered (carry, done). The carry enters at the least
    # significant bit, which is the LAST site, so the chain is built from
    # the right: incoming bond = state after the less significant sites.
    cores = []
    for k in range(nbits):
        if nbits == 1:
            core = np.zeros((1, 2, 2, 1))
            core[0, :, :, 0] = finish
        elif k == nbits - 1:
            # Least significant site: emits (carry, done) to its left.
            core = np.zeros((2, 2, 2, 1))
            core[0, :, :, 0] = carry
            core[1, :, :, 0] = finish
        elif k == 0:
            # Most significant site: accepts done only, so an unresolved
            # carry at the top is dropped, removing wraparound.
            core = np.zeros((1, 2, 2, 2))
            core[0, :, :, 0] = finish
            core[0, :, :, 1] = eye
        else:
            core = np.zeros((2, 2, 2, 2))
            core[0, :, :, 0] = carry
            core[1, :, :, 0] = finish
            core[1, :, :, 1] = eye
        cores.append(core)
    return TTOperator(cores)
