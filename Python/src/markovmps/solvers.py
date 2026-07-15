"""Solvers and the permanent baselines, in one file.

Every production number in this package is produced by a tensor-train
solver and checked against 2 independent references that live beside it:

- The exact engine enumerates the state space, builds the dense generator
  from the model's moves(), and solves it directly: a matrix exponential
  for the survival curve, a linear solve for hitting times and splitting
  probabilities, and the embedded jump chain for the avalanche-size
  distribution. It applies no tensor-train truncation, so a disagreement
  with it isolates a truncation error, and it reaches small systems only.
- Kinetic Monte Carlo samples trajectories of the same process by
  Gillespie stepping, with no operator at all. It is a different method in
  kind, so a disagreement with it at a size both methods reach isolates an
  error in the operator construction, and it reaches system sizes and the
  critical regime the tensor train cannot.

The tensor-train side holds 3 verbs, selected by the observable rather
than one propagator for everything: a linear solve for hitting times,
splitting probabilities, and the avalanche generating function; an
eigensolve for the slowest decay mode of a symmetrisable generator; and a
transient stepper for the full distribution at a finite time. The exact
engine and Monte Carlo build the process from moves() while the
tensor-train verbs build it from terms(), so agreement between the paths
checks the operator assembly itself, not one code path against a copy of
itself.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .tensor_train import DEFAULT_CUTOFF, DEFAULT_MAXDIM, TT, TTOperator

DENSE_STATE_LIMIT = 40_000


# ---------------------------------------------------------------------------
# Exact engine: dense linear algebra on the enumerated state space.
# ---------------------------------------------------------------------------

def exact_generator(model) -> tuple[np.ndarray, list[tuple[int, ...]], dict[tuple[int, ...], int]]:
    """Dense generator from the model's moves(), on the enumerated states.

    Returns
    -------
    (W, states, index)
        W[y, x] is the rate from state x to state y and the diagonal
        carries the escape rates, so every column sums to 0. This path
        never touches the tensor-train operator assembly, which is what
        makes agreement between the two a check of the assembly.

    Raises
    ------
    ValueError
        If the enumerated state space exceeds DENSE_STATE_LIMIT states.
    """
    total = model.state_count()
    if total > DENSE_STATE_LIMIT:
        raise ValueError(
            f"state space of {total} states exceeds the dense limit {DENSE_STATE_LIMIT}"
        )
    states = model.enumerate_states()
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    W = np.zeros((n, n))
    for s in states:
        i = index[s]
        for rate, dest in model.moves(s):
            j = index[dest]
            W[j, i] += rate
            W[i, i] -= rate
    return W, states, index


def exact_survival(model, times: np.ndarray) -> np.ndarray:
    """Survival probability by the dense matrix exponential.

    Survival at time t is the probability mass not yet on the absorbing
    manifold, 1 minus the absorbed mass of exp(W t) applied to the initial
    distribution.
    """
    W, states, index = exact_generator(model)
    times = np.asarray(times, dtype=float)
    p0 = np.zeros(len(states))
    p0[index[model.initial_state()]] = 1.0
    absorbing = np.array([model.is_absorbing(s) for s in states])
    out = np.empty(times.shape)
    for k, t in enumerate(times):
        p = expm(W * t) @ p0
        out[k] = 1.0 - p[absorbing].sum()
    return out


def _transient_split(model, W: np.ndarray, states: list[tuple[int, ...]]):
    """Index masks and blocks used by the dense hitting-time solves."""
    absorbing = np.array([model.is_absorbing(s) for s in states])
    transient = ~absorbing
    Q = W[np.ix_(transient, transient)]
    R = W[np.ix_(absorbing, transient)]
    return absorbing, transient, Q, R


def exact_mfpt(model) -> float:
    """Mean time to absorption from the initial state, by a dense solve.

    Solves the backward system Q^T tau = -1 on the transient block; the
    transpose appears because W is held in the column convention.
    """
    W, states, index = exact_generator(model)
    absorbing, transient, Q, _ = _transient_split(model, W, states)
    tau = np.linalg.solve(Q.T, -np.ones(Q.shape[0]))
    full = np.zeros(len(states))
    full[transient] = tau
    return float(full[index[model.initial_state()]])


def exact_absorption_probability(model, target_predicate) -> float:
    """Probability of absorbing in a chosen part of the absorbing manifold.

    Parameters
    ----------
    target_predicate : callable
        Maps an absorbing state tuple to True when it belongs to the
        target, for example the right-hand barrier of the biased walk.
    """
    W, states, index = exact_generator(model)
    absorbing, transient, Q, R = _transient_split(model, W, states)
    abs_states = [s for s, a in zip(states, absorbing) if a]
    weights = np.array([1.0 if target_predicate(s) else 0.0 for s in abs_states])
    rhs = -(weights @ R)
    h = np.linalg.solve(Q.T, rhs)
    full = np.zeros(len(states))
    full[transient] = h
    for s, a in zip(states, absorbing):
        if a:
            full[index[s]] = 1.0 if target_predicate(s) else 0.0
    return float(full[index[model.initial_state()]])


def exact_avalanche_size_distribution(model, smax: int) -> np.ndarray:
    """Distribution of the number of jumps before absorption, dense engine.

    The jump count is the avalanche size of the Manna model, since every
    jump is one toppling. The embedded jump chain J[y, x] = rate / escape
    is iterated from the initial distribution; the surviving transient
    mass after s jumps gives P(S > s), and the returned array holds
    P(S = s) for s = 0 .. smax to machine precision, including the tail
    that trajectory sampling undersamples.
    """
    if smax < 0:
        raise ValueError("smax must be non-negative")
    W, states, index = exact_generator(model)
    n = len(states)
    absorbing = np.array([model.is_absorbing(s) for s in states])
    J = np.zeros((n, n))
    for s in states:
        i = index[s]
        moves = model.moves(s)
        total = sum(rate for rate, _ in moves)
        for rate, dest in moves:
            J[index[dest], i] += rate / total
    p = np.zeros(n)
    p[index[model.initial_state()]] = 1.0
    dist = np.zeros(smax + 1)
    surviving = p[~absorbing].sum()
    dist[0] = 1.0 - surviving
    for s in range(1, smax + 1):
        p = J @ p
        now = p[~absorbing].sum()
        dist[s] = surviving - now
        surviving = now
    return dist


def exact_avalanche_pgf(model, z: complex) -> complex:
    """Generating function E[z**S] of the jump count, by a dense solve.

    Solves (diag(r) - z T^T) u = z R_in on the transient block with u = 1
    on the absorbing manifold, then contracts with the initial
    distribution. This is the dense twin of tt_avalanche_pgf in the Manna
    model file and shares no code with it.
    """
    W, states, index = exact_generator(model)
    absorbing, transient, Q, R = _transient_split(model, W, states)
    r = -np.diag(Q)
    T_trans = Q + np.diag(r)
    rhs = z * (np.ones(R.shape[0]) @ R)
    A = np.diag(r) - z * T_trans.T
    u = np.linalg.solve(A.astype(complex), rhs.astype(complex))
    full = np.ones(len(states), dtype=complex)
    full[transient] = u
    return complex(full[index[model.initial_state()]])


# ---------------------------------------------------------------------------
# Kinetic Monte Carlo: Gillespie trajectories, no operator at all.
# ---------------------------------------------------------------------------

def kmc_trajectory(model, rng: np.random.Generator,
                   max_jumps: int = 10_000_000) -> tuple[float, int, tuple[int, ...]]:
    """One Gillespie trajectory to absorption.

    Returns
    -------
    (absorption_time, jump_count, final_state)

    Raises
    ------
    RuntimeError
        If the trajectory exceeds max_jumps, which signals a state space
        without reachable absorption rather than a long avalanche.
    """
    state = model.initial_state()
    t = 0.0
    jumps = 0
    while True:
        moves = model.moves(state)
        if not moves:
            return t, jumps, state
        rates = np.array([rate for rate, _ in moves])
        total = rates.sum()
        t += rng.exponential(1.0 / total)
        choice = rng.choice(len(moves), p=rates / total)
        state = moves[choice][1]
        jumps += 1
        if jumps > max_jumps:
            raise RuntimeError(f"trajectory exceeded {max_jumps} jumps without absorbing")


def kmc_survival(model, times: np.ndarray, nshots: int, seed: int) -> np.ndarray:
    """Survival estimate: the fraction of trajectories still active at each time."""
    if nshots < 1:
        raise ValueError("nshots must be at least 1")
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)
    absorbed_at = np.array([kmc_trajectory(model, rng)[0] for _ in range(nshots)])
    return np.array([(absorbed_at > t).mean() for t in times])


def kmc_avalanche_sizes(model, nshots: int, seed: int) -> np.ndarray:
    """Sampled avalanche sizes: jumps to absorption per trajectory."""
    if nshots < 1:
        raise ValueError("nshots must be at least 1")
    rng = np.random.default_rng(seed)
    return np.array([kmc_trajectory(model, rng)[1] for _ in range(nshots)])


def kmc_mfpt(model, nshots: int, seed: int) -> tuple[float, float]:
    """Mean absorption time and its standard error from sampled trajectories."""
    if nshots < 2:
        raise ValueError("nshots must be at least 2 for a standard error")
    rng = np.random.default_rng(seed)
    samples = np.array([kmc_trajectory(model, rng)[0] for _ in range(nshots)])
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(nshots))


# ---------------------------------------------------------------------------
# Tensor-train verbs: the transient stepper, the linear solve, the eigensolve.
# ---------------------------------------------------------------------------

def tt_evolve(generator: TTOperator, p0: TT, tfinal: float, dt: float,
              order: int = 4, cutoff: float = DEFAULT_CUTOFF,
              maxdim: int = DEFAULT_MAXDIM) -> TT:
    """Propagate dp/dt = W p to tfinal by truncated-Taylor steps.

    Each step applies p <- sum_{j=0}^{order} (dt**j / j!) W**j p with
    compression after every operator application. The terms of the
    generator do not commute, so no step is exact: the per-step error is
    O(dt**(order + 1)) and the global error O(dt**order). The step is
    stable when dt times the largest escape rate is small; the
    observable-level wrappers in the model files choose dt from the
    model's rate bound, and a caller supplying dt directly owns that
    choice.

    Parameters
    ----------
    generator : TTOperator
        The full generator W with columns summing to 0.
    p0 : TT
        Initial distribution.
    tfinal : float
        Evolution time; the step count is ceil(tfinal / dt) and the step
        used is tfinal divided by that count, so tfinal is hit exactly.
    dt : float
        Requested step size.
    order : int
        Taylor order per step, at least 1.
    """
    if tfinal < 0:
        raise ValueError("tfinal must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if order < 1:
        raise ValueError("order must be at least 1")
    if tfinal == 0:
        return p0.copy()
    nsteps = max(1, int(np.ceil(tfinal / dt - 1e-12)))
    step = tfinal / nsteps
    p = p0.copy()
    for _ in range(nsteps):
        term = p
        total = p
        for j in range(1, order + 1):
            term = generator.apply(term, cutoff=cutoff, maxdim=maxdim).scale(step / j)
            total = total.add(term).compress(cutoff=cutoff, maxdim=maxdim)
        p = total
    return p


def _mixed_canonical(x: TT) -> TT:
    """Right-canonical copy: orthogonality centre at site 0, ranks kept."""
    cores = [c.copy() for c in x.cores]
    n = len(cores)
    for k in range(n - 1, 0, -1):
        rl, d, rr = cores[k].shape
        m = cores[k].reshape(rl, d * rr)
        q, r = np.linalg.qr(m.conj().T)
        keep = q.shape[1]
        cores[k] = q.conj().T.reshape(keep, d, rr)
        cores[k - 1] = np.tensordot(cores[k - 1], r.conj().T, axes=([2], [0]))
    return TT(cores)


def _local_operator(LA, a1, a2, RA):
    """Dense two-site operator from the environments; row and column order
    is (rank_left, d1, d2, rank_right)."""
    left = np.einsum("xay,aijb->xiyjb", LA, a1, optimize=True)
    right = np.einsum("bklc,ucv->bkulv", a2, RA, optimize=True)
    block = np.einsum("xiyjb,bkulv->xikuyjlv", left, right, optimize=True)
    m = block.shape[0] * block.shape[1] * block.shape[2] * block.shape[3]
    return block.reshape(m, m)


def _local_vector(Lb, b1, b2, Rb):
    """Dense two-site right-hand side from the environments."""
    left = np.einsum("xb,bic->xic", Lb, b1, optimize=True)
    right = np.einsum("cjd,ud->cju", b2, Rb, optimize=True)
    block = np.einsum("xic,cju->xiju", left, right, optimize=True)
    return block.reshape(-1)


def _split_two_site(theta, rl, d1, d2, rr, cutoff, maxdim, direction):
    """SVD split of a two-site block into left and right cores."""
    m = theta.reshape(rl * d1, d2 * rr)
    u, s, vh = np.linalg.svd(m, full_matrices=False)
    snorm = s[0] if s.size and s[0] > 0 else 1.0
    keep = max(1, min(int(np.sum(s > cutoff * snorm)), maxdim))
    u, s, vh = u[:, :keep], s[:keep], vh[:keep]
    if direction == "right":
        left_core = u.reshape(rl, d1, keep)
        right_core = (s[:, None] * vh).reshape(keep, d2, rr)
    else:
        left_core = (u * s).reshape(rl, d1, keep)
        right_core = vh.reshape(keep, d2, rr)
    return left_core, right_core


def tt_linear_solve(A: TTOperator, b: TT, x0: TT | None = None, sweeps: int = 8,
                    cutoff: float = DEFAULT_CUTOFF, maxdim: int = DEFAULT_MAXDIM,
                    tol: float = 1e-10) -> tuple[TT, float]:
    """Solve A x = b in tensor-train form by two-site alternating updates.

    At each pair of neighbouring sites the equation is projected onto the
    frame of the current solution (a Galerkin projection with the trial
    space as the test space) and the small dense system is solved exactly;
    the two-site block is then split by SVD, which is where the solution
    rank adapts. The residual norm(A x - b) / norm(b) is evaluated after
    every full sweep and the iteration stops when it drops below tol.

    The projection solves the local system directly rather than through
    the normal equations, so it does not square the condition number; the
    systems this package produces (hitting times, splitting probabilities,
    the tilted generating function) have dominant diagonals away from the
    critical regime, where the projected system stays well posed. A
    singular local system falls back to least squares.

    Returns
    -------
    (x, residual) : (TT, float)
        The solution and the final relative residual.
    """
    if sweeps < 1:
        raise ValueError("sweeps must be at least 1")
    n = A.nsites
    if n < 2:
        raise ValueError("the two-site solver needs at least 2 sites")
    complex_problem = any(np.iscomplexobj(c) for c in A.cores + b.cores)
    if x0 is None:
        x = b.copy()
    else:
        x = x0.copy()
    if complex_problem:
        x = TT([c.astype(complex) for c in x.cores])
    x = _mixed_canonical(x)
    bnorm = b.norm()
    if bnorm == 0:
        raise ValueError("the right-hand side is zero")

    def env_stacks(x):
        LAs = [np.ones((1, 1, 1), dtype=complex if complex_problem else float)]
        Lbs = [np.ones((1, 1), dtype=complex if complex_problem else float)]
        RAs = [np.ones((1, 1, 1), dtype=complex if complex_problem else float)]
        Rbs = [np.ones((1, 1), dtype=complex if complex_problem else float)]
        for k in range(n - 1, 1, -1):
            RAs.insert(0, np.einsum("xib,aijc,yjd,bcd->xay", x.cores[k].conj(), A.cores[k], x.cores[k], RAs[0], optimize=True))
            Rbs.insert(0, np.einsum("xib,cid,bd->xc", x.cores[k].conj(), b.cores[k], Rbs[0], optimize=True))
        return LAs, Lbs, RAs, Rbs

    LAs, Lbs, RAs, Rbs = env_stacks(x)
    residual = np.inf
    for _ in range(sweeps):
        # Left-to-right half sweep over bonds (k, k+1).
        for k in range(n - 1):
            LA, Lb = LAs[-1], Lbs[-1]
            RA, Rb = RAs[0], Rbs[0]
            a1, a2 = A.cores[k], A.cores[k + 1]
            b1, b2 = b.cores[k], b.cores[k + 1]
            rl = x.cores[k].shape[0]
            rr = x.cores[k + 1].shape[2]
            d1, d2 = a1.shape[2], a2.shape[2]
            Aloc = _local_operator(LA, a1, a2, RA)
            bloc = _local_vector(Lb, b1, b2, Rb)
            try:
                theta = np.linalg.solve(Aloc, bloc)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(Aloc, bloc, rcond=None)[0]
            lc, rc = _split_two_site(theta, rl, d1, d2, rr, cutoff, maxdim, "right")
            x.cores[k], x.cores[k + 1] = lc, rc
            if k < n - 2:
                LAs.append(np.einsum("xay,xib,aijc,yjd->bcd", LAs[-1], lc.conj(), a1, lc, optimize=True))
                Lbs.append(np.einsum("xb,xic,bid->cd", Lbs[-1], lc.conj(), b1, optimize=True))
                RAs.pop(0)
                Rbs.pop(0)
        # Rebuild environments for the return sweep.
        LAs = [np.ones((1, 1, 1), dtype=complex if complex_problem else float)]
        Lbs = [np.ones((1, 1), dtype=complex if complex_problem else float)]
        for k in range(0, n - 2):
            LAs.append(np.einsum("xay,xib,aijc,yjd->bcd", LAs[-1], x.cores[k].conj(), A.cores[k], x.cores[k], optimize=True))
            Lbs.append(np.einsum("xb,xic,bid->cd", Lbs[-1], x.cores[k].conj(), b.cores[k], optimize=True))
        RAs = [np.ones((1, 1, 1), dtype=complex if complex_problem else float)]
        Rbs = [np.ones((1, 1), dtype=complex if complex_problem else float)]
        # Right-to-left half sweep.
        for k in range(n - 2, -1, -1):
            LA, Lb = LAs[k], Lbs[k]
            RA, Rb = RAs[0], Rbs[0]
            a1, a2 = A.cores[k], A.cores[k + 1]
            b1, b2 = b.cores[k], b.cores[k + 1]
            rl = x.cores[k].shape[0]
            rr = x.cores[k + 1].shape[2]
            d1, d2 = a1.shape[2], a2.shape[2]
            Aloc = _local_operator(LA, a1, a2, RA)
            bloc = _local_vector(Lb, b1, b2, Rb)
            try:
                theta = np.linalg.solve(Aloc, bloc)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(Aloc, bloc, rcond=None)[0]
            lc, rc = _split_two_site(theta, rl, d1, d2, rr, cutoff, maxdim, "left")
            x.cores[k], x.cores[k + 1] = lc, rc
            RAs.insert(0, np.einsum("xib,aijc,yjd,bcd->xay", rc.conj(), a2, rc, RAs[0], optimize=True))
            Rbs.insert(0, np.einsum("xib,cid,bd->xc", rc.conj(), b.cores[k + 1], Rbs[0], optimize=True))
        residual = A.apply(x, cutoff=cutoff, maxdim=4 * maxdim).add(b.scale(-1.0)).compress(cutoff=cutoff, maxdim=4 * maxdim).norm() / bnorm
        if residual < tol:
            break
        LAs, Lbs, RAs, Rbs = env_stacks(x)
    return x, float(residual)


def tt_eigen_max(A: TTOperator, x0: TT, sweeps: int = 8,
                 cutoff: float = DEFAULT_CUTOFF, maxdim: int = DEFAULT_MAXDIM) -> tuple[float, TT]:
    """Largest algebraic eigenvalue of a symmetric TT operator, two-site sweeps.

    The operator must be symmetric (the symmetrised generator of a
    detailed-balance model); the local two-site problem is a dense
    symmetric eigenproblem and the top eigenvector is kept. The Manna
    generator is irreversible and has no symmetrisation, so this verb does
    not apply to it; its spectral quantities go through Monte Carlo, the
    stated boundary of the package.

    Returns
    -------
    (eigenvalue, eigenvector) : (float, TT)
    """
    if sweeps < 1:
        raise ValueError("sweeps must be at least 1")
    n = A.nsites
    if n < 2:
        raise ValueError("the two-site solver needs at least 2 sites")
    x = _mixed_canonical(x0.copy())
    value = -np.inf
    for _ in range(sweeps):
        LAs = [np.ones((1, 1, 1))]
        RAs = [np.ones((1, 1, 1))]
        for k in range(n - 1, 1, -1):
            RAs.insert(0, np.einsum("xib,aijc,yjd,bcd->xay", x.cores[k].conj(), A.cores[k], x.cores[k], RAs[0], optimize=True))
        for k in range(n - 1):
            LA = LAs[-1]
            RA = RAs[0]
            a1, a2 = A.cores[k], A.cores[k + 1]
            rl = x.cores[k].shape[0]
            rr = x.cores[k + 1].shape[2]
            d1, d2 = a1.shape[2], a2.shape[2]
            Aloc = _local_operator(LA, a1, a2, RA)
            Aloc = 0.5 * (Aloc + Aloc.T)
            vals, vecs = np.linalg.eigh(Aloc)
            value = float(vals[-1])
            theta = vecs[:, -1]
            lc, rc = _split_two_site(theta, rl, d1, d2, rr, cutoff, maxdim, "right")
            x.cores[k], x.cores[k + 1] = lc, rc
            if k < n - 2:
                LAs.append(np.einsum("xay,xib,aijc,yjd->bcd", LAs[-1], lc.conj(), a1, lc, optimize=True))
                RAs.pop(0)
    nrm = x.norm()
    if nrm > 0:
        x = x.scale(1.0 / nrm)
    return value, x


# ---------------------------------------------------------------------------
# Cross-check helpers.
# ---------------------------------------------------------------------------

def max_abs_deviation(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute elementwise difference between two arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} against {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0
