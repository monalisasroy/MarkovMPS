# BiasedWalk: a walker on a line with 2 absorbing ends, and its closed forms.
#
# A quantity moves up by one at rate rate_up and down by one at rate
# rate_down, on the integer positions 0 through 2^nbits - 1, and the 2 end
# positions absorb. Every observable this model reports has a closed form,
# so the model is the correctness anchor of the package: the splitting
# probability and the mean first-passage time follow from the gambler's
# ruin recurrences, and the survival curve follows from the exact spectrum
# of the symmetrised generator, a tridiagonal matrix with known eigenpairs.
# The tensor-train path encodes the position in binary, one site per bit,
# so the distribution over 2^n positions is a chain of n two-state sites
# and the hop operators are the carry-chain shifts of tensortrain.jl.
#
# Conventions, stated once:
# - Positions run 0 to L with L = 2^nbits - 1; positions 0 and L absorb.
# - rate_up is the rate of the move x to x + 1, rate_down of x to x - 1,
#   both from transient positions only.
# - The bias ratio is rho = rate_down / rate_up; rho = 1 is the unbiased walk.
# - States are held as 1-element vectors [x] so the exact engine and Monte
#   Carlo treat both models through one interface.

struct BiasedWalk
    nbits::Int
    npositions::Int
    last::Int
    rate_up::Float64
    rate_down::Float64
    start::Int
    function BiasedWalk(nbits::Int, rate_up::Real, rate_down::Real, start::Int)
        nbits >= 2 || error("nbits must be at least 2 so a transient region exists")
        (rate_up > 0 && rate_down > 0) || error("rate_up and rate_down must be positive")
        npositions = 2^nbits
        last = npositions - 1
        0 < start < last || error("start must lie strictly between 0 and $last")
        return new(nbits, npositions, last, Float64(rate_up), Float64(rate_down), start)
    end
end

# ---- state-space interface used by the exact engine and Monte Carlo ----

local_dims(w::BiasedWalk) = fill(2, w.nbits)

"Binary digits of a position, most significant bit first."
function state_to_sites(w::BiasedWalk, position::Int)
    0 <= position < w.npositions || error("position $position out of range")
    return [(position >> (w.nbits - k)) & 1 for k in 1:w.nbits]
end

"Number of states, counted without enumerating them, so the dense-limit guard is one comparison rather than the allocation it prevents."
state_count(w::BiasedWalk) = w.npositions

enumerate_states(w::BiasedWalk) = [[x] for x in 0:(w.npositions - 1)]

is_absorbing(w::BiasedWalk, state::Vector{Int}) = state[1] == 0 || state[1] == w.last

"Outgoing moves from a state: (rate, destination) pairs. This is the state-space definition of the process, written without matrices, so the exact engine and Monte Carlo stay independent of the tensor-train operator assembly."
function moves(w::BiasedWalk, state::Vector{Int})
    is_absorbing(w, state) && return Tuple{Float64,Vector{Int}}[]
    x = state[1]
    return [(w.rate_up, [x + 1]), (w.rate_down, [x - 1])]
end

initial_state(w::BiasedWalk) = [w.start]

# ---- tensor-train interface used by the generator assembly ----

"Delta distribution on the starting position, rank 1."
initial_tt(w::BiasedWalk) = product_state(local_dims(w), state_to_sites(w, w.start))

"Diagonal projector onto the transient positions 1 .. L - 1."
function transient_projector_tt(w::BiasedWalk)
    dims = local_dims(w)
    corner0 = product_state(dims, state_to_sites(w, 0))
    cornerL = product_state(dims, state_to_sites(w, w.last))
    indicator = compress(add(add(tt_ones(dims), scale(corner0, -1.0)), scale(cornerL, -1.0)))
    return diagonal_from_tt(indicator)
end

"Diagonal projector onto the 2 absorbing positions."
function absorbing_projector_tt(w::BiasedWalk)
    dims = local_dims(w)
    corner0 = product_state(dims, state_to_sites(w, 0))
    cornerL = product_state(dims, state_to_sites(w, w.last))
    return diagonal_from_tt(compress(add(corner0, cornerL)))
end

"Off-diagonal transition operators: (rate, operator) pairs. The raising and lowering shifts are restricted to moves out of transient positions by the transient projector on the input side; the shifts themselves already drop the moves that leave the line."
function terms(w::BiasedWalk)
    proj = transient_projector_tt(w)
    up = compose(qtt_shift(w.nbits, +1), proj)
    down = compose(qtt_shift(w.nbits, -1), proj)
    return TermSpec[(w.rate_up, up), (w.rate_down, down)]
end

# ---- closed forms: the analytic anchors ----

"Bias ratio rate_down / rate_up."
rho(w::BiasedWalk) = w.rate_down / w.rate_up

"Probability of absorbing at L rather than at 0, from a position. The gambler's ruin form (1 - rho^x) / (1 - rho^L), with the unbiased limit x / L taken exactly when the rates are equal."
function analytic_splitting(w::BiasedWalk, position::Int = w.start)
    0 <= position <= w.last || error("position $position out of range")
    r = rho(w)
    isapprox(r, 1.0) && return position / w.last
    return (1.0 - r^position) / (1.0 - r^w.last)
end

"Mean time to absorption at either end, from a position. For unequal rates the optional-stopping identity on the drift gives (L h(x) - x) / (rate_up - rate_down) with h the splitting probability; for equal rates the limit is x (L - x) / (2 rate)."
function analytic_mfpt(w::BiasedWalk, position::Int = w.start)
    0 <= position <= w.last || error("position $position out of range")
    isapprox(w.rate_up, w.rate_down) && return position * (w.last - position) / (2.0 * w.rate_up)
    h = analytic_splitting(w, position)
    return (w.last * h - position) / (w.rate_up - w.rate_down)
end

"All decay rates of the transient generator, slowest first. The transient block is a constant tridiagonal matrix; conjugating by the detailed-balance weights rho^(x/2) makes it symmetric with off-diagonal sqrt(rate_up rate_down), so its eigenvalues are -(rate_up + rate_down) + 2 sqrt(rate_up rate_down) cos(k pi / L) for k = 1 .. L - 1, and the decay rates are their negatives."
function analytic_decay_rates(w::BiasedWalk)
    geometric = 2.0 * sqrt(w.rate_up * w.rate_down)
    return [(w.rate_up + w.rate_down) - geometric * cos(k * pi / w.last)
            for k in 1:(w.last - 1)]
end

"Survival probability at the given times, from the spectral form. The eigenvectors of the symmetrised transient block are sin(k pi x / L); undoing the similarity weights gives a finite sum that uses no matrix exponential, so it is independent of the dense engine it is checked against."
function analytic_survival(w::BiasedWalk, times::Vector{Float64};
                           position::Int = w.start)
    0 < position < w.last || error("survival is defined from a transient position")
    weights = spectral_weights(w, position)
    rates = analytic_decay_rates(w)
    return [sum(weights .* exp.(-rates .* t)) for t in times]
end

"First-passage time density, minus the derivative of the survival."
function analytic_fpt_density(w::BiasedWalk, times::Vector{Float64};
                              position::Int = w.start)
    0 < position < w.last || error("the density is defined from a transient position")
    weights = spectral_weights(w, position)
    rates = analytic_decay_rates(w)
    return [sum(rates .* weights .* exp.(-rates .* t)) for t in times]
end

"Mode weights of the spectral survival sum: [exp(Q t)](y, x0) summed over transient y, with the detailed-balance similarity undone."
function spectral_weights(w::BiasedWalk, position::Int)
    len = w.last
    r = rho(w)
    xs = 1:(len - 1)
    log_g = isapprox(r, 1.0) ? zeros(len - 1) : [0.5 * x * log(r) for x in xs]
    g_x0 = isapprox(r, 1.0) ? 1.0 : exp(0.5 * position * log(r))
    v = [sqrt(2.0 / len) * sin(pi * k * x / len) for k in 1:(len - 1), x in xs]
    return (v * exp.(-log_g)) .* v[:, position] .* g_x0
end

# ---- tensor-train observables, one solver per observable ----

"Full generator, off-diagonal part, and escape vector for a walk."
function walk_generator(w::BiasedWalk; cutoff::Float64 = 1e-12)
    return markov_generator(local_dims(w), terms(w); cutoff = cutoff)
end

"""
Survival curve by the tensor-train transient stepper.

The step defaults to 0.2 divided by the total hop rate, which keeps the
truncated-Taylor step well inside its stability region; the docstring of
tt_evolve states the step error.
"""
function tt_survival(w::BiasedWalk, times::Vector{Float64};
                     dt::Union{Float64,Nothing} = nothing, order::Int = 4,
                     cutoff::Float64 = 1e-12, maxdim::Int = 64)
    all(times .>= 0) || error("times must be non-negative")
    generator, _, _ = walk_generator(w; cutoff = cutoff)
    step = dt === nothing ? 0.2 / (w.rate_up + w.rate_down) : dt
    absorb = absorbing_projector_tt(w)
    ones_vec = tt_ones(local_dims(w))
    out = zeros(length(times))
    p = initial_tt(w)
    t_now = 0.0
    for k in sortperm(times)
        p = tt_evolve(generator, p, times[k] - t_now, step; order = order,
                      cutoff = cutoff, maxdim = maxdim)
        t_now = times[k]
        absorbed = real(inner(ones_vec, apply(absorb, p; cutoff = cutoff, maxdim = maxdim)))
        out[k] = 1.0 - absorbed
    end
    return out
end

"""
Mean first-passage time by the tensor-train linear solve.

Solves (Pi_abs - W^T) tau = 1_transient, the backward hitting-time system
regularised by the absorbing projector so tau vanishes on the barriers,
then reads tau at the starting position. Returns (mfpt, residual).
"""
function tt_mfpt(w::BiasedWalk; sweeps::Int = 8, cutoff::Float64 = 1e-12,
                 maxdim::Int = 64)
    generator, _, _ = walk_generator(w; cutoff = cutoff)
    absorb = absorbing_projector_tt(w)
    A = compress(add(absorb, scale(tt_transpose(generator), -1.0));
                 cutoff = cutoff, maxdim = OPERATOR_MAXDIM)
    ones_vec = tt_ones(local_dims(w))
    b = compress(add(ones_vec, scale(apply(absorb, ones_vec), -1.0)))
    tau, residual = tt_linear_solve(A, b; sweeps = sweeps, cutoff = cutoff, maxdim = maxdim)
    return real(inner(initial_tt(w), tau)), residual
end

"""
Probability of hitting the far barrier first, by the linear solve.

Solves (Pi_abs - W^T) h = delta_L: the splitting probability is harmonic
for the backward generator, 1 at the far barrier and 0 at the near one.
Returns (probability, residual).
"""
function tt_splitting(w::BiasedWalk; sweeps::Int = 8, cutoff::Float64 = 1e-12,
                      maxdim::Int = 64)
    generator, _, _ = walk_generator(w; cutoff = cutoff)
    absorb = absorbing_projector_tt(w)
    A = compress(add(absorb, scale(tt_transpose(generator), -1.0));
                 cutoff = cutoff, maxdim = OPERATOR_MAXDIM)
    b = product_state(local_dims(w), state_to_sites(w, w.last))
    h, residual = tt_linear_solve(A, b; sweeps = sweeps, cutoff = cutoff, maxdim = maxdim)
    return real(inner(initial_tt(w), h)), residual
end

"""
Slowest decay rate of the transient block, by the symmetrised eigensolve.

The walk obeys detailed balance, so conjugating the generator by the
weights rho^(x/2) makes its transient block symmetric; the projector onto
the transient positions removes the absorbing corner, and the absorbing
subspace is shifted down by shift times the total rate so the spectral
edge of the projected operator is the slowest transient mode. The returned
rate is checked against the closed form
(rate_up + rate_down) - 2 sqrt(rate_up rate_down) cos(pi / L) in the test
suite.
"""
function tt_slowest_decay_rate(w::BiasedWalk; sweeps::Int = 8,
                               cutoff::Float64 = 1e-12, maxdim::Int = 64,
                               shift::Float64 = 10.0)
    generator, _, _ = walk_generator(w; cutoff = cutoff)
    r = rho(w)
    n = w.nbits
    g_mats = Dict{Int,Matrix{Float64}}()
    ginv_mats = Dict{Int,Matrix{Float64}}()
    for k in 1:n
        weight = r^(2.0^(n - k) / 2.0)
        g_mats[k] = [1.0 0.0; 0.0 weight]
        ginv_mats[k] = [1.0 0.0; 0.0 1.0 / weight]
    end
    dims = local_dims(w)
    G = product_operator(dims, g_mats)
    Ginv = product_operator(dims, ginv_mats)
    sym = compose(compose(G, generator), Ginv)
    proj = transient_projector_tt(w)
    projected = compose(compose(proj, sym), proj)
    total_rate = w.rate_up + w.rate_down
    pushed = compress(add(projected, scale(absorbing_projector_tt(w), -shift * total_rate));
                      cutoff = cutoff, maxdim = OPERATOR_MAXDIM)
    seed = compress(apply(proj, tt_ones(dims)))
    value, _ = tt_eigen_max(pushed, seed; sweeps = sweeps, cutoff = cutoff, maxdim = maxdim)
    return -value
end
