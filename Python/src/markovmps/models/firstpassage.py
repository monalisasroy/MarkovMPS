"""BiasedWalk: a walker on a line with two absorbing ends, and its closed forms.

A quantity moves up by one at rate rate_up and down by one at rate
rate_down, on the integer positions 0 through 2**nbits - 1, and the two end
positions absorb. Every observable this model reports has a closed form, so
the model is the correctness anchor of the package: the splitting
probability and the mean first-passage time follow from the gambler's ruin
recurrences, and the survival curve follows from the exact spectrum of the
symmetrised generator, a tridiagonal matrix with known eigenpairs. The
tensor-train path encodes the position in binary, one site per bit, so the
distribution over 2**n positions is a chain of n two-state sites and the
hop operators are the carry-chain shifts of tensor_train.qtt_shift.

Conventions, stated once:

- Positions run 0 to L with L = 2**nbits - 1; positions 0 and L absorb.
- rate_up is the rate of the move x to x + 1, rate_down of x to x - 1,
  both from transient positions only.
- The bias ratio is rho = rate_down / rate_up; rho = 1 is the unbiased walk.
- The closed forms below are standard results for the birth-death chain
  with constant rates; each is re-derived in docs/algorithm.md and checked
  against the dense engine in the test suite before the tensor-train path
  is compared to it.
"""

from __future__ import annotations

import numpy as np

from ..tensor_train import OPERATOR_MAXDIM, TT, TTOperator, qtt_shift


class BiasedWalk:
    """Biased nearest-neighbour walk on 0..2**nbits - 1 with absorbing ends.

    Parameters
    ----------
    nbits : int
        Number of binary sites; positions run 0 to 2**nbits - 1.
    rate_up : float
        Rate of the move x to x + 1 from a transient position.
    rate_down : float
        Rate of the move x to x - 1 from a transient position.
    start : int
        Initial position; must be transient.

    Raises
    ------
    ValueError
        If a rate is not positive, nbits is below 2, or start is not a
        transient position.
    """

    def __init__(self, nbits: int, rate_up: float, rate_down: float, start: int):
        if nbits < 2:
            raise ValueError("nbits must be at least 2 so a transient region exists")
        if rate_up <= 0 or rate_down <= 0:
            raise ValueError("rate_up and rate_down must be positive")
        self.nbits = int(nbits)
        self.npositions = 2 ** self.nbits
        self.last = self.npositions - 1
        if not 0 < start < self.last:
            raise ValueError(f"start must lie strictly between 0 and {self.last}")
        self.rate_up = float(rate_up)
        self.rate_down = float(rate_down)
        self.start = int(start)

    # ---- state-space interface used by the exact engine and Monte Carlo ----

    def local_dims(self) -> list[int]:
        """Local dimensions of the binary-address tensor train."""
        return [2] * self.nbits

    def state_to_sites(self, position: int) -> list[int]:
        """Binary digits of a position, most significant bit first."""
        if not 0 <= position < self.npositions:
            raise ValueError(f"position {position} out of range")
        return [(position >> (self.nbits - 1 - k)) & 1 for k in range(self.nbits)]

    def state_count(self) -> int:
        """Number of states, counted without enumerating them.

        The dense-limit guard compares this before any enumeration, so
        refusing an oversized model costs one comparison, not the
        allocation it exists to prevent.
        """
        return self.npositions

    def enumerate_states(self) -> list[tuple[int, ...]]:
        """All positions, each as a one-element tuple."""
        return [(x,) for x in range(self.npositions)]

    def is_absorbing(self, state: tuple[int, ...]) -> bool:
        """True at the two ends of the line."""
        return state[0] == 0 or state[0] == self.last

    def moves(self, state: tuple[int, ...]) -> list[tuple[float, tuple[int, ...]]]:
        """Outgoing moves from a state: (rate, destination) pairs.

        This is the state-space definition of the process, written without
        matrices, so the exact engine and Monte Carlo stay independent of
        the tensor-train operator assembly.
        """
        (x,) = state
        if self.is_absorbing(state):
            return []
        return [(self.rate_up, (x + 1,)), (self.rate_down, (x - 1,))]

    def initial_state(self) -> tuple[int, ...]:
        """The starting position as a state tuple."""
        return (self.start,)

    # ---- tensor-train interface used by the generator assembly ----

    def initial_tt(self) -> TT:
        """Delta distribution on the starting position, rank 1."""
        return TT.product_state(self.local_dims(), self.state_to_sites(self.start))

    def transient_projector_tt(self) -> TTOperator:
        """Diagonal projector onto the transient positions 1..L-1."""
        dims = self.local_dims()
        ones = TT.ones(dims)
        corner0 = TT.product_state(dims, self.state_to_sites(0))
        cornerL = TT.product_state(dims, self.state_to_sites(self.last))
        indicator = ones.add(corner0.scale(-1.0)).add(cornerL.scale(-1.0)).compress()
        return TTOperator.diagonal_from_tt(indicator)

    def absorbing_projector_tt(self) -> TTOperator:
        """Diagonal projector onto the two absorbing positions."""
        dims = self.local_dims()
        corner0 = TT.product_state(dims, self.state_to_sites(0))
        cornerL = TT.product_state(dims, self.state_to_sites(self.last))
        return TTOperator.diagonal_from_tt(corner0.add(cornerL).compress())

    def terms(self) -> list[tuple[float, TTOperator]]:
        """Off-diagonal transition operators: (rate, operator) pairs.

        The raising and lowering shifts are restricted to moves out of
        transient positions by the transient projector on the input side;
        the shifts themselves already drop the moves that leave the line.
        """
        proj = self.transient_projector_tt()
        up = qtt_shift(self.nbits, +1).compose(proj)
        down = qtt_shift(self.nbits, -1).compose(proj)
        return [(self.rate_up, up), (self.rate_down, down)]

    # ---- closed forms: the analytic anchors ----

    def rho(self) -> float:
        """Bias ratio rate_down / rate_up."""
        return self.rate_down / self.rate_up

    def analytic_splitting(self, position: int | None = None) -> float:
        """Probability of absorbing at L rather than at 0, from a position.

        The gambler's ruin form (1 - rho**x) / (1 - rho**L), with the
        unbiased limit x / L taken exactly when the rates are equal.
        """
        x = self.start if position is None else position
        if not 0 <= x <= self.last:
            raise ValueError(f"position {x} out of range")
        rho = self.rho()
        length = self.last
        if np.isclose(rho, 1.0):
            return x / length
        return float((1.0 - rho ** x) / (1.0 - rho ** length))

    def analytic_mfpt(self, position: int | None = None) -> float:
        """Mean time to absorption at either end, from a position.

        For unequal rates the optional-stopping identity on the drift gives
        (L h(x) - x) / (rate_up - rate_down) with h the splitting
        probability; for equal rates the limit is x (L - x) / (2 rate).
        """
        x = self.start if position is None else position
        if not 0 <= x <= self.last:
            raise ValueError(f"position {x} out of range")
        length = self.last
        if np.isclose(self.rate_up, self.rate_down):
            return float(x * (length - x) / (2.0 * self.rate_up))
        h = self.analytic_splitting(x)
        return float((length * h - x) / (self.rate_up - self.rate_down))

    def analytic_decay_rates(self) -> np.ndarray:
        """All decay rates of the transient generator, slowest first.

        The transient block is a constant tridiagonal matrix; conjugating
        by the detailed-balance weights (rate_down / rate_up)**(x / 2)
        makes it symmetric with off-diagonal sqrt(rate_up * rate_down), so
        its eigenvalues are
        -(rate_up + rate_down) + 2 sqrt(rate_up rate_down) cos(k pi / L)
        for k = 1 .. L - 1, and the decay rates are their negatives.
        """
        length = self.last
        k = np.arange(1, length)
        geo = 2.0 * np.sqrt(self.rate_up * self.rate_down)
        return (self.rate_up + self.rate_down) - geo * np.cos(k * np.pi / length)

    def analytic_survival(self, times: np.ndarray, position: int | None = None) -> np.ndarray:
        """Survival probability at the given times, from the spectral form.

        The eigenvectors of the symmetrised transient block are
        sin(k pi x / L); undoing the similarity weights gives
        S(t | x0) = sum_k exp(-gamma_k t) c_k(x0) sum_y w(y) v_k(y)
        with w the inverse detailed-balance weight. The sum is finite and
        uses no matrix exponential, so it is independent of the dense
        engine it is checked against.
        """
        x0 = self.start if position is None else position
        if not 0 < x0 < self.last:
            raise ValueError("survival is defined from a transient position")
        times = np.asarray(times, dtype=float)
        length = self.last
        xs = np.arange(1, length)
        ks = np.arange(1, length)
        rho = self.rho()
        # Similarity weights g(x) = rho**(x/2): Q = G^{-1} S G with G = diag(g).
        log_g = 0.5 * xs * np.log(rho) if not np.isclose(rho, 1.0) else np.zeros_like(xs, dtype=float)
        v = np.sqrt(2.0 / length) * np.sin(np.pi * np.outer(ks, xs) / length)
        rates = self.analytic_decay_rates()
        # [exp(Q t)]_{y,x0} = g(x0)/g(y) sum_k e^{-gamma_k t} v_k(y) v_k(x0)
        g_x0 = np.exp(0.5 * x0 * np.log(rho)) if not np.isclose(rho, 1.0) else 1.0
        weights = (v @ np.exp(-log_g)) * v[:, x0 - 1] * g_x0
        return np.exp(-np.outer(times, rates)) @ weights

    def analytic_fpt_density(self, times: np.ndarray, position: int | None = None) -> np.ndarray:
        """First-passage time density, minus the derivative of the survival."""
        x0 = self.start if position is None else position
        times = np.asarray(times, dtype=float)
        length = self.last
        xs = np.arange(1, length)
        ks = np.arange(1, length)
        rho = self.rho()
        log_g = 0.5 * xs * np.log(rho) if not np.isclose(rho, 1.0) else np.zeros_like(xs, dtype=float)
        v = np.sqrt(2.0 / length) * np.sin(np.pi * np.outer(ks, xs) / length)
        rates = self.analytic_decay_rates()
        g_x0 = np.exp(0.5 * x0 * np.log(rho)) if not np.isclose(rho, 1.0) else 1.0
        weights = (v @ np.exp(-log_g)) * v[:, x0 - 1] * g_x0
        return np.exp(-np.outer(times, rates)) @ (rates * weights)



# ---------------------------------------------------------------------------
# Tensor-train observables for the biased walk. Each function assembles the
# generator through markov_generator, selects the solver the observable
# needs, and returns the number the closed forms above anchor.
# ---------------------------------------------------------------------------

def walk_generator(walk: BiasedWalk, cutoff: float = 1e-12):
    """Full generator, off-diagonal part, and escape vector for a walk."""
    from ..generator import markov_generator
    return markov_generator(walk.local_dims(), walk.terms(), cutoff=cutoff)


def tt_survival(walk: BiasedWalk, times: np.ndarray, dt: float | None = None,
                order: int = 4, cutoff: float = 1e-12, maxdim: int = 64) -> np.ndarray:
    """Survival curve by the tensor-train transient stepper.

    The step defaults to 0.2 divided by the total hop rate, which keeps
    the truncated-Taylor step well inside its stability region; the
    docstring of solvers.tt_evolve states the step error.
    """
    from ..solvers import tt_evolve
    times = np.asarray(times, dtype=float)
    if np.any(times < 0):
        raise ValueError("times must be non-negative")
    generator, _, _ = walk_generator(walk, cutoff=cutoff)
    total_rate = walk.rate_up + walk.rate_down
    step = 0.2 / total_rate if dt is None else dt
    absorb = walk.absorbing_projector_tt()
    ones = TT.ones(walk.local_dims())
    out = np.empty(times.shape)
    order_idx = np.argsort(times)
    p = walk.initial_tt()
    t_now = 0.0
    for k in order_idx:
        t = times[k]
        p = tt_evolve(generator, p, t - t_now, step, order=order,
                      cutoff=cutoff, maxdim=maxdim)
        t_now = t
        absorbed = ones.inner(absorb.apply(p, cutoff=cutoff, maxdim=maxdim)).real
        out[k] = 1.0 - absorbed
    return out


def tt_mfpt(walk: BiasedWalk, sweeps: int = 8, cutoff: float = 1e-12,
            maxdim: int = 64) -> tuple[float, float]:
    """Mean first-passage time by the tensor-train linear solve.

    Solves (Pi_abs - W^T) tau = 1_transient, the backward hitting-time
    system regularised by the absorbing projector so tau vanishes on the
    barriers, then reads tau at the starting position.

    Returns
    -------
    (mfpt, residual) : (float, float)
        The hitting time and the relative residual of the solve.
    """
    from ..solvers import tt_linear_solve
    generator, _, _ = walk_generator(walk, cutoff=cutoff)
    absorb = walk.absorbing_projector_tt()
    A = absorb.add(generator.transpose().scale(-1.0)).compress(cutoff=cutoff, maxdim=OPERATOR_MAXDIM)
    dims = walk.local_dims()
    ones = TT.ones(dims)
    b = ones.add(absorb.apply(ones).scale(-1.0)).compress()
    tau, residual = tt_linear_solve(A, b, sweeps=sweeps, cutoff=cutoff, maxdim=maxdim)
    value = walk.initial_tt().inner(tau).real
    return float(value), residual


def tt_splitting(walk: BiasedWalk, sweeps: int = 8, cutoff: float = 1e-12,
                 maxdim: int = 64) -> tuple[float, float]:
    """Probability of hitting the far barrier first, by the linear solve.

    Solves (Pi_abs - W^T) h = delta_L: the splitting probability is
    harmonic for the backward generator, 1 at the far barrier and 0 at
    the near one.
    """
    from ..solvers import tt_linear_solve
    generator, _, _ = walk_generator(walk, cutoff=cutoff)
    absorb = walk.absorbing_projector_tt()
    A = absorb.add(generator.transpose().scale(-1.0)).compress(cutoff=cutoff, maxdim=OPERATOR_MAXDIM)
    dims = walk.local_dims()
    b = TT.product_state(dims, walk.state_to_sites(walk.last))
    h, residual = tt_linear_solve(A, b, sweeps=sweeps, cutoff=cutoff, maxdim=maxdim)
    value = walk.initial_tt().inner(h).real
    return float(value), residual


def tt_slowest_decay_rate(walk: BiasedWalk, sweeps: int = 8, cutoff: float = 1e-12,
                          maxdim: int = 64, shift: float = 10.0) -> float:
    """Slowest decay rate of the transient block, by the symmetrised eigensolve.

    The walk obeys detailed balance, so conjugating the generator by the
    weights rho**(x/2) makes its transient block symmetric; the projector
    onto the transient positions removes the absorbing corner, and the
    absorbing subspace is shifted down by shift times the total rate so
    the spectral edge of the projected operator is the slowest transient
    mode. The returned rate is checked against the closed form
    (rate_up + rate_down) - 2 sqrt(rate_up rate_down) cos(pi / L) in the
    test suite.
    """
    from ..solvers import tt_eigen_max
    from ..tensor_train import TTOperator
    generator, _, _ = walk_generator(walk, cutoff=cutoff)
    rho = walk.rho()
    dims = walk.local_dims()
    n = walk.nbits
    g_mats = {}
    ginv_mats = {}
    for k in range(n):
        weight = rho ** (2 ** (n - 1 - k) / 2.0)
        g_mats[k] = np.diag([1.0, weight])
        ginv_mats[k] = np.diag([1.0, 1.0 / weight])
    G = TTOperator.product_operator(dims, g_mats)
    Ginv = TTOperator.product_operator(dims, ginv_mats)
    sym = G.compose(generator).compose(Ginv)
    proj = walk.transient_projector_tt()
    projected = proj.compose(sym).compose(proj)
    total_rate = walk.rate_up + walk.rate_down
    pushed = projected.add(
        walk.absorbing_projector_tt().scale(-shift * total_rate)
    ).compress(cutoff=cutoff, maxdim=OPERATOR_MAXDIM)
    seed = TT.ones(dims)
    seed = proj.apply(seed).compress()
    value, _ = tt_eigen_max(pushed, seed, sweeps=sweeps, cutoff=cutoff, maxdim=maxdim)
    return float(-value)
