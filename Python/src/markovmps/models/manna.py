"""MannaSandpile: the fixed-energy stochastic sandpile and its avalanche observables.

Grains sit on a ring of sites. A site holding 2 or more grains is active
and topples at rate 1: it gives away 2 grains, each sent independently to
its left or right neighbour with probability one half. The total number of
grains never changes, and the dynamics stops when every site holds at most
1 grain, the absorbing manifold. An avalanche started from an active seed
is measured by its size, the total number of topplings before the dynamics
stops, and by its duration; both develop heavy tails as the grain density
approaches the critical value of the conserved (Manna) universality class.

This file holds the model definition and its observables. The size
distribution is computed 3 independent ways in solvers.py: kinetic Monte
Carlo counts topplings per trajectory, the exact engine iterates the
embedded jump chain on the enumerated grain sector, and the tensor-train
path solves 1 linear system per point of the probability generating
function E[z**S], which the escape-diagonal structure of the generator
turns into diag(r) u - z T^T u = boundary, with r the local activity count
and T the toppling operator.

Conventions, stated once:

- Sites are 0-based on a ring of nsites; neighbours wrap around.
- A site is active when its occupation is at least 2.
- Toppling moves are blocked when a destination would exceed nmax; with
  the default nmax = grains the block never engages and the dynamics is
  the untruncated Manna model. A lowered nmax is a stated approximation.
- The initial configuration is deterministic: 2 grains on the seed site,
  then 1 grain per site from site 0 upward, skipping the seed, until the
  grains are placed. This keeps the initial distribution a rank-1 product
  state and makes every run reproducible without a seed.
"""

from __future__ import annotations

import numpy as np

from ..tensor_train import OPERATOR_MAXDIM, TT, TTOperator


class MannaSandpile:
    """Fixed-energy Manna sandpile on a ring.

    Parameters
    ----------
    nsites : int
        Number of lattice sites; at least 3.
    grains : int
        Total number of grains, conserved by the dynamics. Absorbing
        configurations exist only when grains <= nsites, and the seeded
        initial configuration needs grains <= nsites + 1.
    nmax : int or None
        Largest occupation represented; None means grains, which removes
        the truncation entirely.
    seed_site : int or None
        Site given the initial double occupation; None means the middle.

    Raises
    ------
    ValueError
        If the ring is too short, the grain count leaves no absorbing
        configuration, or the seeded initial configuration does not fit.
    """

    def __init__(self, nsites: int, grains: int, nmax: int | None = None,
                 seed_site: int | None = None):
        if nsites < 3:
            raise ValueError("nsites must be at least 3")
        if grains < 2:
            raise ValueError("grains must be at least 2 so the seed site is active")
        if grains > nsites:
            raise ValueError(
                "grains must not exceed nsites, or no absorbing configuration exists"
            )
        self.nsites = int(nsites)
        self.grains = int(grains)
        self.nmax = self.grains if nmax is None else int(nmax)
        if self.nmax < 2:
            raise ValueError("nmax must be at least 2 so a site can be active")
        if self.nmax > self.grains:
            raise ValueError("nmax above the total grain count changes nothing; use nmax = grains")
        self.seed_site = self.nsites // 2 if seed_site is None else int(seed_site)
        if not 0 <= self.seed_site < self.nsites:
            raise ValueError("seed_site out of range")
        self._initial = self._seeded_configuration()

    def _seeded_configuration(self) -> tuple[int, ...]:
        """Deterministic initial configuration: a double seed on a background of 1s."""
        occ = [0] * self.nsites
        occ[self.seed_site] = 2
        remaining = self.grains - 2
        site = 0
        while remaining > 0:
            if site >= self.nsites:
                raise ValueError("grains do not fit the seeded initial configuration")
            if site != self.seed_site:
                occ[site] = 1
                remaining -= 1
            site += 1
        return tuple(occ)

    # ---- state-space interface used by the exact engine and Monte Carlo ----

    def local_dims(self) -> list[int]:
        """Local dimensions of the occupation tensor train."""
        return [self.nmax + 1] * self.nsites

    def state_count(self) -> int:
        """Number of configurations in the conserved sector, without enumerating.

        A dynamic programme over sites counts the compositions of grains
        into nsites occupations each at most nmax, so the dense-limit
        guard can refuse an oversized model before any allocation.
        Python integers do not overflow, so the count is exact at any
        size.
        """
        counts = [0] * (self.grains + 1)
        counts[0] = 1
        for _ in range(self.nsites):
            new = [0] * (self.grains + 1)
            for held, ways in enumerate(counts):
                if ways == 0:
                    continue
                for occ in range(min(self.nmax, self.grains - held) + 1):
                    new[held + occ] += ways
            counts = new
        return counts[self.grains]

    def enumerate_states(self) -> list[tuple[int, ...]]:
        """Every configuration in the conserved grain sector, occupations capped.

        The grain number is conserved, so the generator never leaves this
        sector and the exact engine enumerates only it; the full product
        space would be (nmax + 1)**nsites states for no gain.
        """
        states: list[tuple[int, ...]] = []

        def fill(prefix: list[int], remaining: int) -> None:
            site = len(prefix)
            if site == self.nsites:
                if remaining == 0:
                    states.append(tuple(prefix))
                return
            tail_capacity = (self.nsites - site - 1) * self.nmax
            low = max(0, remaining - tail_capacity)
            high = min(self.nmax, remaining)
            for occ in range(low, high + 1):
                fill(prefix + [occ], remaining - occ)

        fill([], self.grains)
        return states

    def activity(self, state: tuple[int, ...]) -> int:
        """Number of active sites, occupation at least 2."""
        return sum(1 for n in state if n >= 2)

    def is_absorbing(self, state: tuple[int, ...]) -> bool:
        """True when no site is active."""
        return self.activity(state) == 0

    def moves(self, state: tuple[int, ...]) -> list[tuple[float, tuple[int, ...]]]:
        """Outgoing moves from a configuration: (rate, destination) pairs.

        Each active site topples at total rate 1, split over the 3 grain
        destinations: both grains left (rate 1/4), both right (1/4), one
        each way (1/2). A destination that would exceed nmax blocks that
        branch; with nmax = grains no branch is ever blocked.
        """
        out: list[tuple[float, tuple[int, ...]]] = []
        for i, n in enumerate(state):
            if n < 2:
                continue
            left = (i - 1) % self.nsites
            right = (i + 1) % self.nsites
            for rate, dl, dr in ((0.25, 2, 0), (0.25, 0, 2), (0.5, 1, 1)):
                new = list(state)
                new[i] -= 2
                new[left] += dl
                new[right] += dr
                if new[left] > self.nmax or new[right] > self.nmax:
                    continue
                out.append((rate, tuple(new)))
        return out

    def initial_state(self) -> tuple[int, ...]:
        """The seeded configuration."""
        return self._initial

    # ---- tensor-train interface used by the generator assembly ----

    def initial_tt(self) -> TT:
        """Delta distribution on the seeded configuration, rank 1."""
        return TT.product_state(self.local_dims(), list(self._initial))

    def terms(self) -> list[tuple[float, dict[int, np.ndarray]]]:
        """Off-diagonal toppling terms as product operators.

        Each term is (rate, site -> matrix): the toppling matrix at the
        active site and grain-addition matrices at the neighbours. The
        matrices carry the nmax block: an addition that would overflow has
        no matrix entry, which matches the blocked branch in moves() and
        keeps the two constructions the same process.
        """
        d = self.nmax + 1
        topple = np.zeros((d, d))
        for n in range(2, d):
            topple[n - 2, n] = 1.0
        add1 = np.zeros((d, d))
        for n in range(0, d - 1):
            add1[n + 1, n] = 1.0
        add2 = np.zeros((d, d))
        for n in range(0, d - 2):
            add2[n + 2, n] = 1.0
        terms: list[tuple[float, dict[int, np.ndarray]]] = []
        for i in range(self.nsites):
            left = (i - 1) % self.nsites
            right = (i + 1) % self.nsites
            terms.append((0.25, {i: topple, left: add2}))
            terms.append((0.25, {i: topple, right: add2}))
            terms.append((0.5, {i: topple, left: add1, right: add1}))
        return terms

    def activity_tt(self) -> TT:
        """TT vector whose entry at a configuration is its active-site count.

        This is the diagonal of the total toppling rate diag(r); it feeds
        both the escape diagonal check and the generating-function solve.
        """
        dims = self.local_dims()
        d = self.nmax + 1
        active = np.zeros((1, d, 1))
        active[0, 2:, 0] = 1.0
        one = np.ones((1, d, 1))
        total: TT | None = None
        for i in range(self.nsites):
            cores = [active.copy() if k == i else one.copy() for k in range(self.nsites)]
            term = TT(cores)
            total = term if total is None else total.add(term).compress()
        assert total is not None
        return total

    def absorbing_projector_tt(self) -> TTOperator:
        """Diagonal projector onto the absorbing manifold, every site below 2.

        A product operator: the per-site diagonal keeps occupations 0 and
        1, so the projector has rank 1 no matter how large the ring is.
        """
        d = self.nmax + 1
        keep = np.zeros((d, d))
        keep[0, 0] = 1.0
        keep[1, 1] = 1.0
        return TTOperator.product_operator(self.local_dims(), {i: keep for i in range(self.nsites)})

    def grain_sector_projector_tt(self) -> TTOperator:
        """Diagonal projector onto the conserved sector of exactly grains grains.

        The bond index carries the running grain count, capped one above
        the target so any overflow is remembered, giving rank grains + 2.
        The generator conserves the grain number, so conjugating a
        backward system by this projector leaves the physical sector
        untouched while removing the unreachable capped sectors of the
        product space, whose blocked dynamics never stops and would
        otherwise make the mean-size system at z = 1 inconsistent.
        """
        d = self.nmax + 1
        cap = self.grains + 1
        cores = []
        for i in range(self.nsites):
            rl = 1 if i == 0 else cap + 1
            rr = 1 if i == self.nsites - 1 else cap + 1
            core = np.zeros((rl, d, d, rr))
            for c_in in range(rl):
                start = 0 if i == 0 else c_in
                for n in range(d):
                    total = min(start + n, cap)
                    if i == self.nsites - 1:
                        if total == self.grains:
                            core[c_in, n, n, 0] = 1.0
                    else:
                        core[c_in, n, n, total] = 1.0
            cores.append(core)
        return TTOperator(cores)


# ---------------------------------------------------------------------------
# Tensor-train observables for the Manna sandpile. The generating-function
# solve reuses the same linear-solve verb that computes hitting times for
# the biased walk; the survival curve reuses the same transient stepper.
# ---------------------------------------------------------------------------

def sandpile_generator(pile: MannaSandpile, cutoff: float = 1e-12):
    """Full generator, toppling operator, and escape vector for a sandpile."""
    from ..generator import markov_generator
    return markov_generator(pile.local_dims(), pile.terms(), cutoff=cutoff)


def tt_survival(pile: MannaSandpile, times: np.ndarray, dt: float | None = None,
                order: int = 4, cutoff: float = 1e-10, maxdim: int = 64) -> np.ndarray:
    """Probability that the pile is still active at each time.

    The transient stepper propagates the full distribution; the absorbing
    projector, a rank-1 product operator, reads off the stopped mass. The
    step defaults to 0.2 divided by the number of sites, a bound on the
    total toppling rate since each active site topples at rate 1.
    """
    from ..solvers import tt_evolve
    times = np.asarray(times, dtype=float)
    if np.any(times < 0):
        raise ValueError("times must be non-negative")
    generator, _, _ = sandpile_generator(pile, cutoff=cutoff)
    step = 0.2 / pile.nsites if dt is None else dt
    absorb = pile.absorbing_projector_tt()
    ones = TT.ones(pile.local_dims())
    out = np.empty(times.shape)
    order_idx = np.argsort(times)
    p = pile.initial_tt()
    t_now = 0.0
    for k in order_idx:
        t = times[k]
        p = tt_evolve(generator, p, t - t_now, step, order=order,
                      cutoff=cutoff, maxdim=maxdim)
        t_now = t
        absorbed = ones.inner(absorb.apply(p, cutoff=cutoff, maxdim=maxdim)).real
        out[k] = 1.0 - absorbed
    return out


def tt_avalanche_pgf(pile: MannaSandpile, z: complex, sweeps: int = 8,
                     cutoff: float = 1e-10, maxdim: int = 64,
                     warm_start: TT | None = None,
                     assembled: tuple | None = None) -> tuple[complex, float, TT]:
    """Generating function E[z**S] of the avalanche size, by a linear solve.

    S is the number of topplings before the pile stops. The backward
    variable u(x) = E_x[z**S] satisfies diag(r) u - z T^T u = 0 on the
    active configurations with u = 1 on the absorbing manifold, where r is
    the activity count and T the toppling operator; both are low-rank
    tensor trains, so one avalanche moment costs one linear solve and no
    time stepping. Within the conserved grain sector the activity count
    equals the escape rate, and unlike the escape rate it keeps the
    operator nonsingular on the unreachable sectors of the product space
    that the tensor train also spans. The dense twin is
    solvers.exact_avalanche_pgf.

    Parameters
    ----------
    warm_start : TT or None
        Initial guess for the solve. The solution at a neighbouring z is
        a strong guess because E[z**S] varies smoothly in z; the Fourier
        inversion below passes each solution to the next point.
    assembled : tuple or None
        Reuse of (offdiagonal transpose, fixed diagonal part, sector
        projector, right-hand side) across z points, built once by
        assemble_pgf_parts.

    Returns
    -------
    (value, residual, solution) : (complex, float, TT)
        E[z**S] from the seeded initial configuration, the relative
        residual of the solve, and the solution vector for warm starting.
    """
    from ..solvers import tt_linear_solve
    if assembled is None:
        assembled = assemble_pgf_parts(pile, cutoff=cutoff)
    offdiag_t, fixed, _, b = assembled
    A = fixed.add(offdiag_t.scale(-z)).compress(cutoff=cutoff, maxdim=OPERATOR_MAXDIM)
    u, residual = tt_linear_solve(A, b, x0=warm_start, sweeps=sweeps,
                                  cutoff=cutoff, maxdim=maxdim)
    value = pile.initial_tt().inner(u)
    return complex(value), residual, u


def assemble_pgf_parts(pile: MannaSandpile, cutoff: float = 1e-10) -> tuple:
    """Parts of the generating-function system that do not depend on z.

    The fixed part is diag(activity) + Pi_abs + (1 - P_sector): the last
    piece pins the solution to 0 on every grain sector other than the
    conserved one, where the occupation cap creates configurations that
    never stop and would make the z = 1 system inconsistent. The toppling
    operator conserves the grain number, so the physical sector is
    unchanged by the pinning. The right-hand side is projected onto the
    sector for the same reason.
    """
    from ..tensor_train import TTOperator
    _, offdiag, _ = sandpile_generator(pile, cutoff=cutoff)
    offdiag_t = offdiag.transpose()
    rate_diag = TTOperator.diagonal_from_tt(pile.activity_tt())
    absorb = pile.absorbing_projector_tt()
    sector = pile.grain_sector_projector_tt()
    identity = TTOperator.identity(pile.local_dims())
    pin = identity.add(sector.scale(-1.0)).compress(cutoff=cutoff, maxdim=OPERATOR_MAXDIM)
    fixed = rate_diag.add(absorb).add(pin).compress(cutoff=cutoff, maxdim=OPERATOR_MAXDIM)
    b = sector.apply(absorb.apply(TT.ones(pile.local_dims()))).compress()
    return offdiag_t, fixed, sector, b


def tt_mean_avalanche_size(pile: MannaSandpile, sweeps: int = 8,
                           cutoff: float = 1e-10, maxdim: int = 64) -> tuple[float, float]:
    """Mean number of topplings before the pile stops, by a linear solve.

    The mean m(x) = E_x[S] satisfies (diag(r) + Pi_abs - T^T) m = r: one
    more right-hand side for the same operator the generating function
    uses at z = 1.
    """
    from ..solvers import tt_linear_solve
    offdiag_t, fixed, sector, _ = assemble_pgf_parts(pile, cutoff=cutoff)
    A = fixed.add(offdiag_t.scale(-1.0)).compress(cutoff=cutoff, maxdim=OPERATOR_MAXDIM)
    b = sector.apply(pile.activity_tt()).compress()
    m, residual = tt_linear_solve(A, b, sweeps=sweeps, cutoff=cutoff, maxdim=maxdim)
    value = pile.initial_tt().inner(m).real
    return float(value), residual


def tt_avalanche_size_distribution(pile: MannaSandpile, smax: int, npoints: int | None = None,
                                   sweeps: int = 8, cutoff: float = 1e-10,
                                   maxdim: int = 64) -> np.ndarray:
    """Avalanche-size distribution by Fourier inversion of the generating function.

    E[z**S] is evaluated on npoints equally spaced points of the unit
    circle, one tensor-train linear solve per point with the previous
    solution as the warm start, and the distribution is read off by the
    discrete Fourier transform: P(S = s) = (1 / M) sum_j G(z_j) z_j**(-s).
    The inversion aliases the tail: the value returned at s carries the
    mass at s + M, s + 2M and so on, so npoints (default 2 smax + 2)
    bounds the aliasing error by P(S >= npoints), which the caller
    controls through smax. Only the first half of the circle is solved;
    the second half follows from G(conj(z)) = conj(G(z)) because the
    distribution is real.
    """
    if smax < 1:
        raise ValueError("smax must be at least 1")
    M = 2 * smax + 2 if npoints is None else int(npoints)
    if M <= smax:
        raise ValueError("npoints must exceed smax")
    assembled = assemble_pgf_parts(pile, cutoff=cutoff)
    values = np.empty(M, dtype=complex)
    guess: TT | None = None
    for j in range(M // 2 + 1):
        z = np.exp(2j * np.pi * j / M)
        values[j], _, guess = tt_avalanche_pgf(
            pile, z, sweeps=sweeps, cutoff=cutoff, maxdim=maxdim,
            warm_start=guess, assembled=assembled,
        )
    for j in range(M // 2 + 1, M):
        values[j] = np.conj(values[M - j])
    dist = np.fft.fft(values).real / M
    return dist[: smax + 1]
