# Solvers and the permanent baselines, in one file.
#
# Every production number in this package is produced by a tensor-train
# solver and checked against 2 independent references that live beside it:
#
# - The exact engine enumerates the state space, builds the dense generator
#   from the model's moves(), and solves it directly: a matrix exponential
#   for the survival curve, a linear solve for hitting times and splitting
#   probabilities, and the embedded jump chain for the avalanche-size
#   distribution. It applies no tensor-train truncation, so a disagreement
#   with it isolates a truncation error, and it reaches small systems only.
# - Kinetic Monte Carlo samples trajectories of the same process by
#   Gillespie stepping, with no operator at all. It is a different method
#   in kind, so a disagreement with it at a size both methods reach
#   isolates an error in the operator construction, and it reaches system
#   sizes and the critical regime the tensor train cannot.
#
# The tensor-train side holds 3 verbs, selected by the observable rather
# than one propagator for everything: a linear solve for hitting times,
# splitting probabilities, and the avalanche generating function; an
# eigensolve for the slowest decay mode of a symmetrisable generator; and
# a transient stepper for the full distribution at a finite time. The
# exact engine and Monte Carlo build the process from moves() while the
# tensor-train verbs build it from terms(), so agreement between the paths
# checks the operator assembly itself, not one code path against a copy of
# itself.

using Random

const DENSE_STATE_LIMIT = 40_000

# ---------------------------------------------------------------------------
# Exact engine: dense linear algebra on the enumerated state space.
# ---------------------------------------------------------------------------

"Dense generator from the model's moves(), on the enumerated states. Returns (W, states, index); W[y, x] is the rate from state x to state y and the diagonal carries the escape rates, so every column sums to 0. This path never touches the tensor-train operator assembly, which is what makes agreement between the 2 a check of the assembly."
function exact_generator(model)
    total = state_count(model)
    total <= DENSE_STATE_LIMIT ||
        error("state space of $total states exceeds the dense limit $DENSE_STATE_LIMIT")
    states = enumerate_states(model)
    index = Dict{Vector{Int},Int}(s => i for (i, s) in enumerate(states))
    n = length(states)
    W = zeros(n, n)
    for s in states
        i = index[s]
        for (rate, dest) in moves(model, s)
            j = index[dest]
            W[j, i] += rate
            W[i, i] -= rate
        end
    end
    return W, states, index
end

"Survival probability by the dense matrix exponential: 1 minus the absorbed mass of exp(W t) applied to the initial distribution."
function exact_survival(model, times::Vector{Float64})
    W, states, index = exact_generator(model)
    p0 = zeros(length(states))
    p0[index[initial_state(model)]] = 1.0
    absorbing = [is_absorbing(model, s) for s in states]
    return [1.0 - sum((exp(W * t) * p0)[absorbing]) for t in times]
end

"Index masks and blocks used by the dense hitting-time solves."
function transient_split(model, W::Matrix{Float64}, states)
    absorbing = [is_absorbing(model, s) for s in states]
    transient = .!absorbing
    Q = W[transient, transient]
    R = W[absorbing, transient]
    return absorbing, transient, Q, R
end

"Mean time to absorption from the initial state, by a dense solve of the backward system Q^T tau = -1 on the transient block; the transpose appears because W is held in the column convention."
function exact_mfpt(model)
    W, states, index = exact_generator(model)
    absorbing, transient, Q, _ = transient_split(model, W, states)
    tau = transpose(Q) \ (-ones(size(Q, 1)))
    full = zeros(length(states))
    full[transient] = tau
    return full[index[initial_state(model)]]
end

"Probability of absorbing in a chosen part of the absorbing manifold; target_predicate maps an absorbing state to true when it belongs to the target."
function exact_absorption_probability(model, target_predicate)
    W, states, index = exact_generator(model)
    absorbing, transient, Q, R = transient_split(model, W, states)
    abs_states = [s for (s, a) in zip(states, absorbing) if a]
    weights = [target_predicate(s) ? 1.0 : 0.0 for s in abs_states]
    h = transpose(Q) \ (-(transpose(R) * weights))
    full = zeros(length(states))
    full[transient] = h
    for (s, a) in zip(states, absorbing)
        a && (full[index[s]] = target_predicate(s) ? 1.0 : 0.0)
    end
    return full[index[initial_state(model)]]
end

"Distribution of the number of jumps before absorption, dense engine. The embedded jump chain J[y, x] = rate / escape is iterated from the initial distribution; the surviving transient mass after s jumps gives P(S > s), and the returned array holds P(S = s) for s = 0 .. smax to machine precision, including the tail that trajectory sampling undersamples."
function exact_avalanche_size_distribution(model, smax::Int)
    smax >= 0 || error("smax must be non-negative")
    W, states, index = exact_generator(model)
    n = length(states)
    absorbing = [is_absorbing(model, s) for s in states]
    J = zeros(n, n)
    for s in states
        i = index[s]
        outgoing = moves(model, s)
        total = sum(rate for (rate, _) in outgoing; init = 0.0)
        for (rate, dest) in outgoing
            J[index[dest], i] += rate / total
        end
    end
    p = zeros(n)
    p[index[initial_state(model)]] = 1.0
    dist = zeros(smax + 1)
    surviving = sum(p[.!absorbing])
    dist[1] = 1.0 - surviving
    for s in 1:smax
        p = J * p
        now = sum(p[.!absorbing])
        dist[s + 1] = surviving - now
        surviving = now
    end
    return dist
end

"Generating function E[z^S] of the jump count, by a dense solve of (diag(r) - z T^T) u = z R_in on the transient block with u = 1 on the absorbing manifold. This is the dense twin of tt_avalanche_pgf and shares no code with it."
function exact_avalanche_pgf(model, z::Number)
    W, states, index = exact_generator(model)
    absorbing, transient, Q, R = transient_split(model, W, states)
    r = -diag(Q)
    T_trans = Q + Diagonal(r)
    rhs = z .* vec(transpose(ones(size(R, 1))) * R)
    A = Diagonal(complex.(r)) - z .* transpose(complex.(T_trans))
    u = A \ complex.(rhs)
    full = ones(ComplexF64, length(states))
    full[transient] = u
    return full[index[initial_state(model)]]
end

# ---------------------------------------------------------------------------
# Kinetic Monte Carlo: Gillespie trajectories, no operator at all.
# ---------------------------------------------------------------------------

"One Gillespie trajectory to absorption. Returns (absorption_time, jump_count, final_state). Exceeding max_jumps signals a state space without reachable absorption rather than a long avalanche."
function kmc_trajectory(model, rng::AbstractRNG; max_jumps::Int = 10_000_000)
    state = initial_state(model)
    t = 0.0
    jumps = 0
    while true
        outgoing = moves(model, state)
        isempty(outgoing) && return t, jumps, state
        rates = [rate for (rate, _) in outgoing]
        total = sum(rates)
        t += -log(rand(rng)) / total
        draw = rand(rng) * total
        chosen = length(outgoing)
        running = 0.0
        for (k, rate) in enumerate(rates)
            running += rate
            if draw <= running
                chosen = k
                break
            end
        end
        state = outgoing[chosen][2]
        jumps += 1
        jumps <= max_jumps || error("trajectory exceeded $max_jumps jumps without absorbing")
    end
end

"Survival estimate: the fraction of trajectories still active at each time."
function kmc_survival(model, times::Vector{Float64}, nshots::Int, seed::Int)
    nshots >= 1 || error("nshots must be at least 1")
    rng = Xoshiro(seed)
    absorbed_at = [kmc_trajectory(model, rng)[1] for _ in 1:nshots]
    return [count(>(t), absorbed_at) / nshots for t in times]
end

"Sampled avalanche sizes: jumps to absorption per trajectory."
function kmc_avalanche_sizes(model, nshots::Int, seed::Int)
    nshots >= 1 || error("nshots must be at least 1")
    rng = Xoshiro(seed)
    return [kmc_trajectory(model, rng)[2] for _ in 1:nshots]
end

"Mean absorption time and its standard error from sampled trajectories."
function kmc_mfpt(model, nshots::Int, seed::Int)
    nshots >= 2 || error("nshots must be at least 2 for a standard error")
    rng = Xoshiro(seed)
    samples = [kmc_trajectory(model, rng)[1] for _ in 1:nshots]
    mean = sum(samples) / nshots
    variance = sum((samples .- mean) .^ 2) / (nshots - 1)
    return mean, sqrt(variance / nshots)
end

"Largest absolute elementwise difference between 2 arrays."
function max_abs_deviation(a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || error("shape mismatch: $(size(a)) against $(size(b))")
    isempty(a) && return 0.0
    return maximum(abs.(a .- b))
end

# ---------------------------------------------------------------------------
# Tensor-train verbs: the transient stepper, the linear solve, the eigensolve.
# Every contraction below names its index equation in a comment; the Python
# twin computes the same equations through numpy.einsum, and the 2 paths are
# held to the same closed forms and dense references in the test suites.
# ---------------------------------------------------------------------------

"""
Propagate dp/dt = W p to tfinal by truncated-Taylor steps.

Each step applies p <- sum_{j=0}^{order} (dt^j / j!) W^j p with compression
after every operator application. The terms of the generator do not
commute, so no step is exact: the per-step error is O(dt^(order + 1)) and
the global error O(dt^order). The step is stable when dt times the largest
escape rate is small; the observable-level wrappers in the model files
choose dt from the model's rate bound, and a caller supplying dt directly
owns that choice. The step count is ceil(tfinal / dt) and the step used is
tfinal divided by that count, so tfinal is hit exactly.
"""
function tt_evolve(generator::TTop, p0::TTvec, tfinal::Float64, dt::Float64;
                   order::Int = 4, cutoff::Float64 = DEFAULT_CUTOFF,
                   maxdim::Int = DEFAULT_MAXDIM)
    tfinal >= 0 || error("tfinal must be non-negative")
    dt > 0 || error("dt must be positive")
    order >= 1 || error("order must be at least 1")
    tfinal == 0 && return copy(p0)
    nsteps = max(1, Int(ceil(tfinal / dt - 1e-12)))
    step = tfinal / nsteps
    p = copy(p0)
    for _ in 1:nsteps
        term = p
        total = p
        for j in 1:order
            term = scale(apply(generator, term; cutoff = cutoff, maxdim = maxdim), step / j)
            total = compress(add(total, term); cutoff = cutoff, maxdim = maxdim)
        end
        p = total
    end
    return p
end

"Right-canonical copy: orthogonality centre at site 1, ranks kept."
function mixed_canonical(x::TTvec{T}) where {T}
    cores = [copy(c) for c in x.cores]
    n = length(cores)
    for k in n:-1:2
        rl, d, rr = size(cores[k])
        m = reshape(cores[k], rl, d * rr)
        f = qr(Matrix(adjoint(m)))
        q = Matrix(f.Q)
        keep = size(q, 2)
        cores[k] = reshape(Matrix(adjoint(q)), keep, d, rr)
        carry = Matrix(adjoint(f.R))
        prev = cores[k - 1]
        p1, pd, _ = size(prev)
        cores[k - 1] = reshape(reshape(prev, p1 * pd, rl) * carry, p1, pd, keep)
    end
    return TTvec(cores)
end

"Grow a left operator environment by one site: env'[x2, a2, y2] = sum conj(X[x, i, x2]) A[a, i, j, a2] Y[y, j, y2] env[x, a, y]."
function grow_left_op(env::Array{T,3}, X::Array{<:Number,3}, A::Array{<:Number,4},
                      Y::Array{<:Number,3}) where {T}
    rx, ra, ry = size(env)
    _, dj, ry2 = size(Y)
    # T1[x, a, j, y2] = sum_y env[x, a, y] Y[y, j, y2]
    t1 = reshape(reshape(env, rx * ra, ry) * reshape(Y, ry, dj * ry2), rx, ra, dj, ry2)
    ra_a, di, _, ra2 = size(A)
    # T2[x, y2, i, a2] = sum_{a, j} T1[x, a, j, y2] A[a, i, j, a2]
    t1p = reshape(permutedims(t1, (1, 4, 2, 3)), rx * ry2, ra_a * dj)
    ap = reshape(permutedims(A, (1, 3, 2, 4)), ra_a * dj, di * ra2)
    t2 = reshape(t1p * ap, rx, ry2, di, ra2)
    # env'[x2, a2, y2] = sum_{x, i} conj(X[x, i, x2]) T2[x, y2, i, a2]
    rxx, dxi, rx2 = size(X)
    t2p = reshape(permutedims(t2, (1, 3, 4, 2)), rxx * dxi, ra2 * ry2)
    xr = reshape(X, rxx * dxi, rx2)
    return reshape(adjoint(xr) * t2p, rx2, ra2, ry2)
end

"Grow a right operator environment by one site: env'[x, a, y] = sum conj(X[x, i, x2]) A[a, i, j, a2] Y[y, j, y2] env[x2, a2, y2]."
function grow_right_op(X::Array{<:Number,3}, A::Array{<:Number,4},
                       Y::Array{<:Number,3}, env::Array{T,3}) where {T}
    rx2, ra2, ry2 = size(env)
    ry, dj, _ = size(Y)
    # T1[y, j, x2, a2] = sum_{y2} Y[y, j, y2] env[x2, a2, y2]
    envp = reshape(permutedims(env, (3, 1, 2)), ry2, rx2 * ra2)
    t1 = reshape(reshape(Y, ry * dj, ry2) * envp, ry, dj, rx2, ra2)
    ra, di, _, _ = size(A)
    # T2[i, a, y, x2] = sum_{j, a2} A[a, i, j, a2] T1[y, j, x2, a2]
    ap = reshape(permutedims(A, (2, 1, 3, 4)), di * ra, dj * ra2)
    t1p = reshape(permutedims(t1, (2, 4, 1, 3)), dj * ra2, ry * rx2)
    t2 = reshape(ap * t1p, di, ra, ry, rx2)
    # env'[x, a, y] = sum_{i, x2} conj(X[x, i, x2]) T2[i, a, y, x2]
    rxx, dxi, rxx2 = size(X)
    t2p = reshape(permutedims(t2, (1, 4, 2, 3)), dxi * rxx2, ra * ry)
    xp = reshape(permutedims(X, (2, 3, 1)), dxi * rxx2, rxx)
    return reshape(adjoint(xp) * t2p, rxx, ra, ry)
end

"Grow a left vector environment by one site: env'[x2, b2] = sum conj(X[x, i, x2]) B[b, i, b2] env[x, b]."
function grow_left_vec(env::Array{T,2}, X::Array{<:Number,3}, B::Array{<:Number,3}) where {T}
    rx, di, rx2 = size(X)
    rb, _, rb2 = size(B)
    # T[i, x2, b] = sum_x conj(X[x, i, x2]) env[x, b]
    t = reshape(transpose(conj.(reshape(X, rx, di * rx2))) * env, di, rx2, rb)
    # env'[x2, b2] = sum_{i, b} T[i, x2, b] B[b, i, b2]
    tp = reshape(permutedims(t, (2, 1, 3)), rx2, di * rb)
    bp = reshape(permutedims(B, (2, 1, 3)), di * rb, rb2)
    return tp * bp
end

"Grow a right vector environment by one site: env'[x, b] = sum conj(X[x, i, x2]) B[b, i, b2] env[x2, b2]."
function grow_right_vec(X::Array{<:Number,3}, B::Array{<:Number,3}, env::Array{T,2}) where {T}
    rb, di, rb2 = size(B)
    rx, _, rx2 = size(X)
    # T[b, i, x2] = sum_{b2} B[b, i, b2] env[x2, b2]
    t = reshape(reshape(B, rb * di, rb2) * transpose(env), rb, di, rx2)
    # env'[x, b] = sum_{i, x2} conj(X[x, i, x2]) T[b, i, x2]
    xr = conj.(reshape(X, rx, di * rx2))
    tp = reshape(permutedims(t, (2, 3, 1)), di * rx2, rb)
    return xr * tp
end

"Dense 2-site operator from the environments; row and column indices fuse as (rank_left, d1, d2, rank_right) with rank_left fastest, the same fusion split_two_site uses."
function local_operator(LA::Array{<:Number,3}, A1::Array{<:Number,4},
                        A2::Array{<:Number,4}, RA::Array{<:Number,3})
    rx, ra, ry = size(LA)
    _, i1, j1, rb = size(A1)
    _, k1, l1, rc = size(A2)
    ru, _, rv = size(RA)
    # L2[x, y, i, j, b] = sum_a LA[x, a, y] A1[a, i, j, b]
    lap = reshape(permutedims(LA, (1, 3, 2)), rx * ry, ra)
    l2 = reshape(lap * reshape(A1, ra, i1 * j1 * rb), rx, ry, i1, j1, rb)
    # R2[b, k, l, u, v] = sum_c A2[b, k, l, c] RA[u, c, v]
    rap = reshape(permutedims(RA, (2, 1, 3)), rc, ru * rv)
    r2 = reshape(reshape(A2, rb * k1 * l1, rc) * rap, rb, k1, l1, ru, rv)
    # block[x, i, y, j, k, l, u, v] = sum_b L2[x, i, y, j, b] R2[b, k, l, u, v]
    l2p = reshape(permutedims(l2, (1, 3, 2, 4, 5)), rx * i1 * ry * j1, rb)
    block = reshape(l2p * reshape(r2, rb, k1 * l1 * ru * rv), rx, i1, ry, j1, k1, l1, ru, rv)
    # Rows (x, i, k, u), columns (y, j, l, v), each with the first fastest.
    ordered = permutedims(block, (1, 2, 5, 7, 3, 4, 6, 8))
    m = rx * i1 * k1 * ru
    return reshape(ordered, m, ry * j1 * l1 * rv)
end

"Dense 2-site right-hand side from the environments, fused (rank_left, d1, d2, rank_right) with rank_left fastest."
function local_rhs(Lb::Array{<:Number,2}, b1::Array{<:Number,3},
                   b2::Array{<:Number,3}, Rb::Array{<:Number,2})
    rx, rc = size(Lb)
    _, i1, rc2 = size(b1)
    _, k1, rc3 = size(b2)
    ru, _ = size(Rb)
    t1 = reshape(Lb * reshape(b1, rc, i1 * rc2), rx, i1, rc2)
    t2 = reshape(reshape(t1, rx * i1, rc2) * reshape(b2, rc2, k1 * rc3), rx, i1, k1, rc3)
    v = reshape(t2, rx * i1 * k1, rc3) * transpose(Rb)
    return vec(v)
end

"SVD split of a 2-site block into left and right cores; the orthogonality centre moves in the named direction."
function split_two_site(theta::Vector{<:Number}, rl::Int, d1::Int, d2::Int, rr::Int,
                        cutoff::Float64, maxdim::Int, direction::Symbol)
    m = reshape(theta, rl * d1, d2 * rr)
    f = svd(m)
    keep = keep_count(f.S, cutoff, maxdim)
    u = f.U[:, 1:keep]
    s = f.S[1:keep]
    vt = f.Vt[1:keep, :]
    if direction == :right
        left_core = reshape(u, rl, d1, keep)
        right_core = reshape(Diagonal(s) * vt, keep, d2, rr)
    else
        left_core = reshape(u * Diagonal(s), rl, d1, keep)
        right_core = reshape(vt, keep, d2, rr)
    end
    return left_core, right_core
end

"""
Solve A x = b in tensor-train form by 2-site alternating updates.

At each pair of neighbouring sites the equation is projected onto the
frame of the current solution (a Galerkin projection with the trial space
as the test space) and the small dense system is solved exactly; the
2-site block is then split by SVD, which is where the solution rank
adapts. The residual norm(A x - b) / norm(b) is evaluated after every full
sweep and the iteration stops when it drops below tol.

The projection solves the local system directly rather than through the
normal equations, so it does not square the condition number; the systems
this package produces (hitting times, splitting probabilities, the tilted
generating function) have dominant diagonals away from the critical
regime, where the projected system stays well posed. A singular local
system falls back to a column-pivoted least-squares solve.

Returns (x, residual): the solution and the final relative residual.
"""
function tt_linear_solve(A::TTop, b::TTvec; x0 = nothing, sweeps::Int = 8,
                         cutoff::Float64 = DEFAULT_CUTOFF,
                         maxdim::Int = DEFAULT_MAXDIM, tol::Float64 = 1e-10)
    sweeps >= 1 || error("sweeps must be at least 1")
    n = nsites(A)
    n >= 2 || error("the 2-site solver needs at least 2 sites")
    T = promote_type(eltype(A.cores[1]), eltype(b.cores[1]),
                     x0 === nothing ? Float64 : eltype(x0.cores[1]))
    Ac = [Array{T,4}(c) for c in A.cores]
    bc = [Array{T,3}(c) for c in b.cores]
    x = x0 === nothing ? TTvec([Array{T,3}(c) for c in b.cores]) :
                         TTvec([Array{T,3}(c) for c in x0.cores])
    x = mixed_canonical(x)
    bnorm = tt_norm(b)
    bnorm > 0 || error("the right-hand side is zero")

    function solve_local(Aloc, bloc)
        return try
            Aloc \ bloc
        catch err
            err isa LinearAlgebra.SingularException || err isa LinearAlgebra.LAPACKException || rethrow()
            qr(Aloc, ColumnNorm()) \ bloc
        end
    end

    function right_stacks(xv)
        RAs = [ones(T, 1, 1, 1)]
        Rbs = [ones(T, 1, 1)]
        for k in n:-1:3
            pushfirst!(RAs, grow_right_op(xv.cores[k], Ac[k], xv.cores[k], RAs[1]))
            pushfirst!(Rbs, grow_right_vec(xv.cores[k], bc[k], Rbs[1]))
        end
        return RAs, Rbs
    end

    residual = Inf
    for _ in 1:sweeps
        LAs = [ones(T, 1, 1, 1)]
        Lbs = [ones(T, 1, 1)]
        RAs, Rbs = right_stacks(x)
        # Left-to-right half sweep over bonds (k, k + 1).
        for k in 1:(n - 1)
            rl = size(x.cores[k], 1)
            rr = size(x.cores[k + 1], 3)
            d1 = size(Ac[k], 3)
            d2 = size(Ac[k + 1], 3)
            Aloc = local_operator(LAs[end], Ac[k], Ac[k + 1], RAs[1])
            bloc = local_rhs(Lbs[end], bc[k], bc[k + 1], Rbs[1])
            theta = solve_local(Aloc, bloc)
            lc, rc = split_two_site(theta, rl, d1, d2, rr, cutoff, maxdim, :right)
            x.cores[k], x.cores[k + 1] = lc, rc
            if k < n - 1
                push!(LAs, grow_left_op(LAs[end], lc, Ac[k], lc))
                push!(Lbs, grow_left_vec(Lbs[end], lc, bc[k]))
                popfirst!(RAs)
                popfirst!(Rbs)
            end
        end
        # Rebuild the left environments for the return sweep.
        LAs = [ones(T, 1, 1, 1)]
        Lbs = [ones(T, 1, 1)]
        for k in 1:(n - 2)
            push!(LAs, grow_left_op(LAs[end], x.cores[k], Ac[k], x.cores[k]))
            push!(Lbs, grow_left_vec(Lbs[end], x.cores[k], bc[k]))
        end
        RAs = [ones(T, 1, 1, 1)]
        Rbs = [ones(T, 1, 1)]
        # Right-to-left half sweep.
        for k in (n - 1):-1:1
            rl = size(x.cores[k], 1)
            rr = size(x.cores[k + 1], 3)
            d1 = size(Ac[k], 3)
            d2 = size(Ac[k + 1], 3)
            Aloc = local_operator(LAs[k], Ac[k], Ac[k + 1], RAs[1])
            bloc = local_rhs(Lbs[k], bc[k], bc[k + 1], Rbs[1])
            theta = solve_local(Aloc, bloc)
            lc, rc = split_two_site(theta, rl, d1, d2, rr, cutoff, maxdim, :left)
            x.cores[k], x.cores[k + 1] = lc, rc
            pushfirst!(RAs, grow_right_op(rc, Ac[k + 1], rc, RAs[1]))
            pushfirst!(Rbs, grow_right_vec(rc, bc[k + 1], Rbs[1]))
        end
        Ax = apply(TTop(Ac), x; cutoff = cutoff, maxdim = 4 * maxdim)
        residual = tt_norm(compress(add(Ax, scale(b, -1.0));
                                    cutoff = cutoff, maxdim = 4 * maxdim)) / bnorm
        residual < tol && break
    end
    return x, residual
end

"""
Largest algebraic eigenvalue of a symmetric TT operator, 2-site sweeps.

The operator must be symmetric (the symmetrised generator of a
detailed-balance model); the local 2-site problem is a dense symmetric
eigenproblem and the top eigenvector is kept. The Manna generator is
irreversible and has no symmetrisation, so this verb does not apply to it;
its spectral quantities go through Monte Carlo, the stated boundary of the
package. Returns (eigenvalue, eigenvector).
"""
function tt_eigen_max(A::TTop, x0::TTvec; sweeps::Int = 8,
                      cutoff::Float64 = DEFAULT_CUTOFF, maxdim::Int = DEFAULT_MAXDIM)
    sweeps >= 1 || error("sweeps must be at least 1")
    n = nsites(A)
    n >= 2 || error("the 2-site solver needs at least 2 sites")
    x = mixed_canonical(copy(x0))
    value = -Inf
    for _ in 1:sweeps
        LAs = [ones(1, 1, 1)]
        RAs = [ones(1, 1, 1)]
        for k in n:-1:3
            pushfirst!(RAs, grow_right_op(x.cores[k], A.cores[k], x.cores[k], RAs[1]))
        end
        for k in 1:(n - 1)
            rl = size(x.cores[k], 1)
            rr = size(x.cores[k + 1], 3)
            d1 = size(A.cores[k], 3)
            d2 = size(A.cores[k + 1], 3)
            Aloc = local_operator(LAs[end], A.cores[k], A.cores[k + 1], RAs[1])
            Aloc = (Aloc + transpose(Aloc)) / 2
            decomposition = eigen(Symmetric(real.(Aloc)))
            value = decomposition.values[end]
            theta = decomposition.vectors[:, end]
            lc, rc = split_two_site(theta, rl, d1, d2, rr, cutoff, maxdim, :right)
            x.cores[k], x.cores[k + 1] = lc, rc
            if k < n - 1
                push!(LAs, grow_left_op(LAs[end], lc, A.cores[k], lc))
                popfirst!(RAs)
            end
        end
    end
    nrm = tt_norm(x)
    nrm > 0 && (x = scale(x, 1.0 / nrm))
    return value, x
end
