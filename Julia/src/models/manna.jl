# MannaSandpile: the fixed-energy stochastic sandpile and its avalanche observables.
#
# Grains sit on a ring of sites. A site holding 2 or more grains is active
# and topples at rate 1: it gives away 2 grains, each sent independently to
# its left or right neighbour with probability one half. The total number
# of grains never changes, and the dynamics stops when every site holds at
# most 1 grain, the absorbing manifold. An avalanche started from an active
# seed is measured by its size, the total number of topplings before the
# dynamics stops, and by its duration; both develop heavy tails as the
# grain density approaches the critical value of the conserved (Manna)
# universality class.
#
# The size distribution is computed 3 independent ways: kinetic Monte Carlo
# counts topplings per trajectory, the exact engine iterates the embedded
# jump chain on the enumerated grain sector, and the tensor-train path
# solves 1 linear system per point of the probability generating function
# E[z^S].
#
# Conventions, stated once:
# - Sites are 1-based on a ring of nsites; neighbours wrap around.
# - A site is active when its occupation is at least 2.
# - Toppling moves are blocked when a destination would exceed nmax; with
#   the default nmax = grains the block never engages and the dynamics is
#   the untruncated Manna model. A lowered nmax is a stated approximation.
# - The initial configuration is deterministic: 2 grains on the seed site,
#   then 1 grain per site from site 1 upward, skipping the seed, until the
#   grains are placed. This keeps the initial distribution a rank-1 product
#   state and makes every run reproducible without a seed.

struct MannaSandpile
    nsites::Int
    grains::Int
    nmax::Int
    seed_site::Int
    initial::Vector{Int}
    function MannaSandpile(nsites::Int, grains::Int;
                           nmax::Union{Int,Nothing} = nothing,
                           seed_site::Union{Int,Nothing} = nothing)
        nsites >= 3 || error("nsites must be at least 3")
        grains >= 2 || error("grains must be at least 2 so the seed site is active")
        grains <= nsites || error("grains must not exceed nsites, or no absorbing configuration exists")
        cap = nmax === nothing ? grains : nmax
        cap >= 2 || error("nmax must be at least 2 so a site can be active")
        cap <= grains || error("nmax above the total grain count changes nothing; use nmax = grains")
        seed = seed_site === nothing ? div(nsites, 2) + 1 : seed_site
        1 <= seed <= nsites || error("seed_site out of range")
        occupation = zeros(Int, nsites)
        occupation[seed] = 2
        remaining = grains - 2
        site = 1
        while remaining > 0
            site <= nsites || error("grains do not fit the seeded initial configuration")
            if site != seed
                occupation[site] = 1
                remaining -= 1
            end
            site += 1
        end
        return new(nsites, grains, cap, seed, occupation)
    end
end

# ---- state-space interface used by the exact engine and Monte Carlo ----

local_dims(p::MannaSandpile) = fill(p.nmax + 1, p.nsites)

"Number of configurations in the conserved sector, counted without enumerating: a dynamic programme over sites counts the compositions of grains into nsites occupations each at most nmax. BigInt arithmetic keeps the count exact where Int64 would silently wrap on large models, which is exactly where the guard matters."
function state_count(p::MannaSandpile)
    counts = zeros(BigInt, p.grains + 1)
    counts[1] = 1
    for _ in 1:p.nsites
        fresh = zeros(BigInt, p.grains + 1)
        for held in 0:p.grains
            counts[held + 1] == 0 && continue
            for occ in 0:min(p.nmax, p.grains - held)
                fresh[held + occ + 1] += counts[held + 1]
            end
        end
        counts = fresh
    end
    return counts[p.grains + 1]
end

"Every configuration in the conserved grain sector, occupations capped. The grain number is conserved, so the generator never leaves this sector and the exact engine enumerates only it."
function enumerate_states(p::MannaSandpile)
    states = Vector{Int}[]
    function fill_from(prefix::Vector{Int}, remaining::Int)
        site = length(prefix)
        if site == p.nsites
            remaining == 0 && push!(states, copy(prefix))
            return
        end
        tail_capacity = (p.nsites - site - 1) * p.nmax
        low = max(0, remaining - tail_capacity)
        high = min(p.nmax, remaining)
        for occ in low:high
            push!(prefix, occ)
            fill_from(prefix, remaining - occ)
            pop!(prefix)
        end
    end
    fill_from(Int[], p.grains)
    return states
end

"Number of active sites, occupation at least 2."
activity(p::MannaSandpile, state::Vector{Int}) = count(n -> n >= 2, state)

is_absorbing(p::MannaSandpile, state::Vector{Int}) = activity(p, state) == 0

"Outgoing moves from a configuration: (rate, destination) pairs. Each active site topples at total rate 1, split over the 3 grain destinations: both grains left (rate 1/4), both right (1/4), one each way (1/2). A destination that would exceed nmax blocks that branch; with nmax = grains no branch is ever blocked."
function moves(p::MannaSandpile, state::Vector{Int})
    out = Tuple{Float64,Vector{Int}}[]
    for i in 1:p.nsites
        state[i] >= 2 || continue
        left = mod1(i - 1, p.nsites)
        right = mod1(i + 1, p.nsites)
        for (rate, dl, dr) in ((0.25, 2, 0), (0.25, 0, 2), (0.5, 1, 1))
            new = copy(state)
            new[i] -= 2
            new[left] += dl
            new[right] += dr
            (new[left] > p.nmax || new[right] > p.nmax) && continue
            push!(out, (rate, new))
        end
    end
    return out
end

initial_state(p::MannaSandpile) = copy(p.initial)

# ---- tensor-train interface used by the generator assembly ----

"Delta distribution on the seeded configuration, rank 1."
initial_tt(p::MannaSandpile) = product_state(local_dims(p), p.initial)

"Off-diagonal toppling terms as product operators. Each term is (rate, site => matrix): the toppling matrix at the active site and grain-addition matrices at the neighbours. The matrices carry the nmax block: an addition that would overflow has no matrix entry, which matches the blocked branch in moves() and keeps the 2 constructions the same process."
function terms(p::MannaSandpile)
    d = p.nmax + 1
    topple = zeros(d, d)
    for n in 2:(d - 1)
        topple[n - 1, n + 1] = 1.0
    end
    add1 = zeros(d, d)
    for n in 0:(d - 2)
        add1[n + 2, n + 1] = 1.0
    end
    add2 = zeros(d, d)
    for n in 0:(d - 3)
        add2[n + 3, n + 1] = 1.0
    end
    out = TermSpec[]
    for i in 1:p.nsites
        left = mod1(i - 1, p.nsites)
        right = mod1(i + 1, p.nsites)
        push!(out, (0.25, Dict(i => topple, left => add2)))
        push!(out, (0.25, Dict(i => topple, right => add2)))
        push!(out, (0.5, Dict(i => topple, left => add1, right => add1)))
    end
    return out
end

"TT vector whose entry at a configuration is its active-site count. This is the diagonal of the total toppling rate diag(r); it feeds both the escape diagonal check and the generating-function solve."
function activity_tt(p::MannaSandpile)
    dims = local_dims(p)
    d = p.nmax + 1
    active = zeros(1, d, 1)
    active[1, 3:d, 1] .= 1.0
    one_core = ones(1, d, 1)
    total = nothing
    for i in 1:p.nsites
        cores = [k == i ? copy(active) : copy(one_core) for k in 1:p.nsites]
        term = TTvec(cores)
        total = total === nothing ? term : compress(add(total, term))
    end
    return total
end

"Diagonal projector onto the absorbing manifold, every site below 2. A product operator: the per-site diagonal keeps occupations 0 and 1, so the projector has rank 1 no matter how large the ring is."
function absorbing_projector_tt(p::MannaSandpile)
    d = p.nmax + 1
    keep = zeros(d, d)
    keep[1, 1] = 1.0
    keep[2, 2] = 1.0
    return product_operator(local_dims(p), Dict(i => keep for i in 1:p.nsites))
end

"""
Diagonal projector onto the conserved sector of exactly grains grains.

The bond index carries the running grain count, capped one above the
target so any overflow is remembered, giving rank grains + 2. The
generator conserves the grain number, so conjugating a backward system by
this projector leaves the physical sector untouched while removing the
unreachable capped sectors of the product space, whose blocked dynamics
never stops and would otherwise make the mean-size system at z = 1
inconsistent.
"""
function grain_sector_projector_tt(p::MannaSandpile)
    d = p.nmax + 1
    cap = p.grains + 1
    cores = Array{Float64,4}[]
    for i in 1:p.nsites
        rl = i == 1 ? 1 : cap + 1
        rr = i == p.nsites ? 1 : cap + 1
        core = zeros(rl, d, d, rr)
        for c_in in 1:rl
            start = i == 1 ? 0 : c_in - 1
            for n in 0:(d - 1)
                total = min(start + n, cap)
                if i == p.nsites
                    total == p.grains && (core[c_in, n + 1, n + 1, 1] = 1.0)
                else
                    core[c_in, n + 1, n + 1, total + 1] = 1.0
                end
            end
        end
        push!(cores, core)
    end
    return TTop(cores)
end

# ---- tensor-train observables ----

"Full generator, toppling operator, and escape vector for a sandpile."
function sandpile_generator(p::MannaSandpile; cutoff::Float64 = 1e-10)
    return markov_generator(local_dims(p), terms(p); cutoff = cutoff)
end

"""
Probability that the pile is still active at each time.

The transient stepper propagates the full distribution; the absorbing
projector, a rank-1 product operator, reads off the stopped mass. The step
defaults to 0.2 divided by the number of sites, a bound on the total
toppling rate since each active site topples at rate 1.
"""
function tt_survival(p::MannaSandpile, times::Vector{Float64};
                     dt::Union{Float64,Nothing} = nothing, order::Int = 4,
                     cutoff::Float64 = 1e-10, maxdim::Int = 64)
    all(times .>= 0) || error("times must be non-negative")
    generator, _, _ = sandpile_generator(p; cutoff = cutoff)
    step = dt === nothing ? 0.2 / p.nsites : dt
    absorb = absorbing_projector_tt(p)
    ones_vec = tt_ones(local_dims(p))
    out = zeros(length(times))
    state = initial_tt(p)
    t_now = 0.0
    for k in sortperm(times)
        state = tt_evolve(generator, state, times[k] - t_now, step; order = order,
                          cutoff = cutoff, maxdim = maxdim)
        t_now = times[k]
        absorbed = real(inner(ones_vec, apply(absorb, state; cutoff = cutoff, maxdim = maxdim)))
        out[k] = 1.0 - absorbed
    end
    return out
end

"Parts of the generating-function system that do not depend on z. The fixed part is diag(activity) + Pi_abs + (1 - P_sector): the last piece pins the solution to 0 on every grain sector other than the conserved one, where the occupation cap creates configurations that never stop and would make the z = 1 system inconsistent. The toppling operator conserves the grain number, so the physical sector is unchanged by the pinning. The right-hand side is projected onto the sector for the same reason."
function assemble_pgf_parts(p::MannaSandpile; cutoff::Float64 = 1e-10)
    _, offdiag, _ = sandpile_generator(p; cutoff = cutoff)
    offdiag_t = tt_transpose(offdiag)
    rate_diag = diagonal_from_tt(activity_tt(p))
    absorb = absorbing_projector_tt(p)
    sector = grain_sector_projector_tt(p)
    pin = compress(add(tt_identity(local_dims(p)), scale(sector, -1.0));
                   cutoff = cutoff, maxdim = OPERATOR_MAXDIM)
    fixed = compress(add(add(rate_diag, absorb), pin); cutoff = cutoff, maxdim = OPERATOR_MAXDIM)
    b = compress(apply(sector, apply(absorb, tt_ones(local_dims(p)))))
    return offdiag_t, fixed, sector, b
end

"""
Generating function E[z^S] of the avalanche size, by a linear solve.

S is the number of topplings before the pile stops. The backward variable
u(x) = E_x[z^S] satisfies diag(r) u - z T^T u = 0 on the active
configurations with u = 1 on the absorbing manifold, where r is the
activity count and T the toppling operator; both are low-rank tensor
trains, so one avalanche moment costs one linear solve and no time
stepping. Within the conserved grain sector the activity count equals the
escape rate, and unlike the escape rate it keeps the operator nonsingular
on the unreachable sectors of the product space that the tensor train also
spans. The dense twin is exact_avalanche_pgf. The solution at a
neighbouring z is a strong warm start because E[z^S] varies smoothly in z.
Returns (value, residual, solution).
"""
function tt_avalanche_pgf(p::MannaSandpile, z::Number; sweeps::Int = 8,
                          cutoff::Float64 = 1e-10, maxdim::Int = 64,
                          warm_start = nothing, assembled = nothing)
    parts = assembled === nothing ? assemble_pgf_parts(p; cutoff = cutoff) : assembled
    offdiag_t, fixed, _, b = parts
    A = compress(add(fixed, scale(offdiag_t, -z)); cutoff = cutoff, maxdim = OPERATOR_MAXDIM)
    u, residual = tt_linear_solve(A, b; x0 = warm_start, sweeps = sweeps,
                                  cutoff = cutoff, maxdim = maxdim)
    return inner(initial_tt(p), u), residual, u
end

"""
Mean number of topplings before the pile stops, by a linear solve.

The mean m(x) = E_x[S] satisfies (diag(r) + Pi_abs - T^T) m = r: one more
right-hand side for the same operator the generating function uses at
z = 1. Returns (mean, residual).
"""
function tt_mean_avalanche_size(p::MannaSandpile; sweeps::Int = 8,
                                cutoff::Float64 = 1e-10, maxdim::Int = 64)
    offdiag_t, fixed, sector, _ = assemble_pgf_parts(p; cutoff = cutoff)
    A = compress(add(fixed, scale(offdiag_t, -1.0)); cutoff = cutoff, maxdim = OPERATOR_MAXDIM)
    b = compress(apply(sector, activity_tt(p)))
    m, residual = tt_linear_solve(A, b; sweeps = sweeps, cutoff = cutoff, maxdim = maxdim)
    return real(inner(initial_tt(p), m)), residual
end

"""
Avalanche-size distribution by Fourier inversion of the generating function.

E[z^S] is evaluated on npoints equally spaced points of the unit circle,
one tensor-train linear solve per point with the previous solution as the
warm start, and the distribution is read off by the discrete Fourier sum
P(S = s) = (1 / M) sum_j G(z_j) z_j^(-s), evaluated directly since M stays
small. The inversion aliases the tail: the value returned at s carries the
mass at s + M, s + 2M and so on, so npoints (default 2 smax + 2) bounds
the aliasing error by P(S >= npoints), which the caller controls through
smax. Only the first half of the circle is solved; the second half follows
from G(conj(z)) = conj(G(z)) because the distribution is real.
"""
function tt_avalanche_size_distribution(p::MannaSandpile, smax::Int;
                                        npoints::Union{Int,Nothing} = nothing,
                                        sweeps::Int = 8, cutoff::Float64 = 1e-10,
                                        maxdim::Int = 64)
    smax >= 1 || error("smax must be at least 1")
    M = npoints === nothing ? 2 * smax + 2 : npoints
    M > smax || error("npoints must exceed smax")
    parts = assemble_pgf_parts(p; cutoff = cutoff)
    values = zeros(ComplexF64, M)
    guess = nothing
    for j in 0:div(M, 2)
        z = exp(2im * pi * j / M)
        value, _, guess = tt_avalanche_pgf(p, z; sweeps = sweeps, cutoff = cutoff,
                                           maxdim = maxdim, warm_start = guess,
                                           assembled = parts)
        values[j + 1] = value
    end
    for j in (div(M, 2) + 1):(M - 1)
        values[j + 1] = conj(values[M - j + 1])
    end
    dist = [real(sum(values[j + 1] * exp(-2im * pi * j * s / M) for j in 0:(M - 1))) / M
            for s in 0:smax]
    return dist
end
