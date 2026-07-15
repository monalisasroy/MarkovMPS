# Tensor-train vectors and operators for classical probability distributions.
#
# A distribution over N sites with local dimension d has d^N entries; the
# tensor train stores it as N cores of size (rank_left, d, rank_right), at a
# cost set by the correlations in the distribution rather than the size of
# the state space. This file is the dependency-free core: the vector type,
# the operator type, the compression sweep, and the binary-address shift
# operators the first-passage model uses. Nothing here knows about Markov
# generators; that assembly lives in generator.jl. The Python tree holds a
# line-for-line twin of this file.
#
# Conventions, stated once for the whole package:
# - A vector core has size (rank_left, d, rank_right); boundary ranks are 1.
# - An operator core has size (rank_left, d_out, d_in, rank_right).
# - Operator entries follow the column convention of the generator: entry
#   W[y, x] is the rate from state x to state y, so dp/dt = W p and
#   probability conservation reads: every column sums to 0.
# - Dense indexing puts the first site slowest, matching product_state.

using LinearAlgebra

const DEFAULT_CUTOFF = 1e-12
const DEFAULT_MAXDIM = 256
# Rank caps apply to solution tensors only. An operator or right-hand side
# compressed under a rank cap is a different system, which a solver can then
# satisfy to machine precision while answering the wrong question, so system
# objects are compressed by the cutoff alone under this effectively
# unbounded cap.
const OPERATOR_MAXDIM = 2^30

struct TTvec{T<:Number}
    cores::Vector{Array{T,3}}
    function TTvec(cores::Vector{Array{T,3}}) where {T<:Number}
        isempty(cores) && error("a TT vector needs at least one core")
        size(cores[1], 1) == 1 || error("the left boundary rank must be 1")
        size(cores[end], 3) == 1 || error("the right boundary rank must be 1")
        for k in 1:(length(cores) - 1)
            size(cores[k], 3) == size(cores[k + 1], 1) ||
                error("rank mismatch between cores $k and $(k + 1)")
        end
        return new{T}(cores)
    end
end

struct TTop{T<:Number}
    cores::Vector{Array{T,4}}
    function TTop(cores::Vector{Array{T,4}}) where {T<:Number}
        isempty(cores) && error("a TT operator needs at least one core")
        size(cores[1], 1) == 1 || error("the left boundary rank must be 1")
        size(cores[end], 4) == 1 || error("the right boundary rank must be 1")
        for k in 1:(length(cores) - 1)
            size(cores[k], 4) == size(cores[k + 1], 1) ||
                error("rank mismatch between cores $k and $(k + 1)")
        end
        return new{T}(cores)
    end
end

nsites(v::TTvec) = length(v.cores)
nsites(o::TTop) = length(o.cores)
localdims(v::TTvec) = [size(c, 2) for c in v.cores]
dims_out(o::TTop) = [size(c, 2) for c in o.cores]
dims_in(o::TTop) = [size(c, 3) for c in o.cores]
ranks(v::TTvec) = vcat([size(v.cores[1], 1)], [size(c, 3) for c in v.cores])
ranks(o::TTop) = vcat([size(o.cores[1], 1)], [size(c, 4) for c in o.cores])
Base.copy(v::TTvec) = TTvec([copy(c) for c in v.cores])
Base.copy(o::TTop) = TTop([copy(c) for c in o.cores])

"Rank-1 delta distribution on one configuration; values are 0-based to match the state tuples."
function product_state(dims::Vector{Int}, values::Vector{Int})
    length(dims) == length(values) || error("dims and values must have the same length")
    cores = Array{Float64,3}[]
    for (d, v) in zip(dims, values)
        0 <= v < d || error("value $v out of range for local dimension $d")
        core = zeros(1, d, 1)
        core[1, v + 1, 1] = 1.0
        push!(cores, core)
    end
    return TTvec(cores)
end

"Rank-1 all-ones vector, used to sum a distribution's entries."
tt_ones(dims::Vector{Int}) = TTvec([ones(1, d, 1) for d in dims])

"Contract to a dense vector, first site slowest; only for small state spaces."
function to_dense(v::TTvec{T}) where {T}
    total = prod(localdims(v))
    total <= 2_000_000 || error("state space of size $total is too large to densify")
    acc = reshape(v.cores[1], size(v.cores[1], 2), size(v.cores[1], 3))
    for core in v.cores[2:end]
        r, d, rr = size(core)
        prefix = size(acc, 1)
        # acc[(p), r] core[r, i, r'] -> new[(p, i fastest? no: i fast within
        # the appended position), r']: first site slowest means the new site
        # index runs fastest, so permute i ahead of p before fusing.
        m = acc * reshape(core, r, d * rr)                    # (p, (i, r')) i fastest
        m = reshape(m, prefix, d, rr)
        acc = reshape(permutedims(m, (2, 1, 3)), prefix * d, rr)
    end
    return vec(acc)
end

"Exact TT decomposition of a dense vector, first site slowest, by sequential SVD."
function from_dense(vector::Vector{T}, dims::Vector{Int};
                    cutoff::Float64 = 0.0, maxdim::Int = DEFAULT_MAXDIM) where {T}
    total = prod(dims)
    length(vector) == total || error("vector length does not match dims product")
    cores = Array{T,3}[]
    # Reorder to last-site-slowest so column-major unfolding peels the first site.
    tensor = permutedims(reshape(vector, reverse(dims)...), length(dims):-1:1)
    rest = reshape(tensor, dims[1], :)
    rank = 1
    for (k, d) in enumerate(dims[1:(end - 1)])
        m = reshape(rest, rank * d, :)
        f = svd(m)
        keep = keep_count(f.S, cutoff, maxdim)
        push!(cores, reshape(f.U[:, 1:keep], rank, d, keep))
        rest = Diagonal(f.S[1:keep]) * f.Vt[1:keep, :]
        rank = keep
    end
    push!(cores, reshape(rest, rank, dims[end], 1))
    return TTvec(cores)
end

"Number of singular values kept under a relative cutoff and a hard cap."
function keep_count(s::Vector{<:Real}, cutoff::Float64, maxdim::Int)
    maxdim >= 1 || error("maxdim must be at least 1")
    isempty(s) && return 1
    keep = count(x -> x > cutoff * s[1], s)
    return max(1, min(keep, maxdim))
end

"Return the vector multiplied by a scalar."
function scale(v::TTvec, factor::Number)
    cores = [copy(c) for c in v.cores]
    T = promote_type(eltype(cores[1]), typeof(factor))
    out = [Array{T,3}(c) for c in cores]
    out[1] = out[1] .* factor
    return TTvec(out)
end

"Direct-sum addition of 2 TT vectors; compress afterwards to control the rank."
function add(a::TTvec, b::TTvec)
    localdims(a) == localdims(b) || error("cannot add TT vectors with different local dimensions")
    n = nsites(a)
    T = promote_type(eltype(a.cores[1]), eltype(b.cores[1]))
    n == 1 && return TTvec([Array{T,3}(a.cores[1] .+ b.cores[1])])
    cores = Array{T,3}[]
    for k in 1:n
        ca, cb = a.cores[k], b.cores[k]
        d = size(ca, 2)
        if k == 1
            core = zeros(T, 1, d, size(ca, 3) + size(cb, 3))
            core[:, :, 1:size(ca, 3)] = ca
            core[:, :, (size(ca, 3) + 1):end] = cb
        elseif k == n
            core = zeros(T, size(ca, 1) + size(cb, 1), d, 1)
            core[1:size(ca, 1), :, :] = ca
            core[(size(ca, 1) + 1):end, :, :] = cb
        else
            core = zeros(T, size(ca, 1) + size(cb, 1), d, size(ca, 3) + size(cb, 3))
            core[1:size(ca, 1), :, 1:size(ca, 3)] = ca
            core[(size(ca, 1) + 1):end, :, (size(ca, 3) + 1):end] = cb
        end
        push!(cores, core)
    end
    return TTvec(cores)
end

"Euclidean inner product sum(conj(a) .* b)."
function inner(a::TTvec, b::TTvec)
    localdims(a) == localdims(b) || error("cannot contract TT vectors with different local dimensions")
    T = promote_type(eltype(a.cores[1]), eltype(b.cores[1]))
    env = ones(T, 1, 1)
    for (ca, cb) in zip(a.cores, b.cores)
        ra, d, ra2 = size(ca)
        rb, _, rb2 = size(cb)
        # env[a, b] conj(A[a, i, c]) B[b, i, e] -> env[c, e]
        t = transpose(conj.(reshape(ca, ra, d * ra2))) * env       # ((i, c), b)
        t = reshape(t, d, ra2, rb)
        t = reshape(permutedims(t, (2, 1, 3)), ra2, d * rb)        # (c, (i, b))
        bm = reshape(permutedims(cb, (2, 1, 3)), d * rb, rb2)      # ((i, b), e)
        env = t * bm
    end
    return env[1, 1]
end

"Sum of all entries: the total probability of a distribution."
sum_entries(v::TTvec) = inner(tt_ones(localdims(v)), v)

"Euclidean norm."
tt_norm(v::TTvec) = sqrt(abs(inner(v, v)))

"Compressed copy: QR sweep left to right, then an SVD sweep back. Singular values below cutoff times the largest at each bond are dropped, and at most maxdim survive per bond."
function compress(v::TTvec{T}; cutoff::Float64 = DEFAULT_CUTOFF,
                  maxdim::Int = DEFAULT_MAXDIM) where {T}
    cores = [copy(c) for c in v.cores]
    n = length(cores)
    for k in 1:(n - 1)
        rl, d, rr = size(cores[k])
        f = qr(reshape(cores[k], rl * d, rr))
        q = Matrix(f.Q)
        cores[k] = reshape(q, rl, d, size(q, 2))
        r = f.R
        nxt = cores[k + 1]
        r2, d2, rr2 = size(nxt)
        cores[k + 1] = reshape(r * reshape(nxt, r2, d2 * rr2), size(q, 2), d2, rr2)
    end
    for k in n:-1:2
        rl, d, rr = size(cores[k])
        f = svd(reshape(cores[k], rl, d * rr))
        keep = keep_count(f.S, cutoff, maxdim)
        cores[k] = reshape(f.Vt[1:keep, :], keep, d, rr)
        carry = f.U[:, 1:keep] * Diagonal(f.S[1:keep])
        prev = cores[k - 1]
        p1, pd, _ = size(prev)
        cores[k - 1] = reshape(reshape(prev, p1 * pd, rl) * carry, p1, pd, keep)
    end
    return TTvec(cores)
end

"Sum a list of TT vectors, compressing after each addition."
function tt_sum(vectors::Vector{<:TTvec}; cutoff::Float64 = DEFAULT_CUTOFF,
                maxdim::Int = DEFAULT_MAXDIM)
    isempty(vectors) && error("tt_sum needs at least one vector")
    out = vectors[1]
    for v in vectors[2:end]
        out = compress(add(out, v); cutoff = cutoff, maxdim = maxdim)
    end
    return out
end

# --- Operator side ---

"Rank-1 identity operator."
tt_identity(dims::Vector{Int}) = TTop([reshape(Matrix{Float64}(I, d, d), 1, d, d, 1) for d in dims])

"Rank-1 operator: the given matrix at each named site (1-based), identity elsewhere."
function product_operator(dims::Vector{Int}, site_matrices::Dict{Int,Matrix{Float64}})
    for site in keys(site_matrices)
        1 <= site <= length(dims) || error("site index $site out of range for $(length(dims)) sites")
    end
    cores = Array{Float64,4}[]
    for (i, d) in enumerate(dims)
        m = get(site_matrices, i, Matrix{Float64}(I, d, d))
        size(m) == (d, d) || error("matrix at site $i has size $(size(m)), expected ($d, $d)")
        push!(cores, reshape(m, 1, d, d, 1))
    end
    return TTop(cores)
end

"Diagonal operator whose diagonal is the given TT vector."
function diagonal_from_tt(v::TTvec{T}) where {T}
    cores = Array{T,4}[]
    for c in v.cores
        rl, d, rr = size(c)
        core = zeros(T, rl, d, d, rr)
        for i in 1:d
            core[:, i, i, :] = c[:, i, :]
        end
        push!(cores, core)
    end
    return TTop(cores)
end

"Operator transpose: swap the input and output legs of every core."
tt_transpose(o::TTop) = TTop([permutedims(c, (1, 3, 2, 4)) for c in o.cores])

"Return the operator multiplied by a scalar."
function scale(o::TTop, factor::Number)
    T = promote_type(eltype(o.cores[1]), typeof(factor))
    cores = [Array{T,4}(c) for c in o.cores]
    cores[1] = cores[1] .* factor
    return TTop(cores)
end

"Direct-sum addition of 2 TT operators; compress afterwards to control the rank."
function add(a::TTop, b::TTop)
    (dims_in(a) == dims_in(b) && dims_out(a) == dims_out(b)) ||
        error("cannot add TT operators with different local dimensions")
    n = nsites(a)
    T = promote_type(eltype(a.cores[1]), eltype(b.cores[1]))
    n == 1 && return TTop([Array{T,4}(a.cores[1] .+ b.cores[1])])
    cores = Array{T,4}[]
    for k in 1:n
        ca, cb = a.cores[k], b.cores[k]
        dout, din = size(ca, 2), size(ca, 3)
        if k == 1
            core = zeros(T, 1, dout, din, size(ca, 4) + size(cb, 4))
            core[:, :, :, 1:size(ca, 4)] = ca
            core[:, :, :, (size(ca, 4) + 1):end] = cb
        elseif k == n
            core = zeros(T, size(ca, 1) + size(cb, 1), dout, din, 1)
            core[1:size(ca, 1), :, :, :] = ca
            core[(size(ca, 1) + 1):end, :, :, :] = cb
        else
            core = zeros(T, size(ca, 1) + size(cb, 1), dout, din, size(ca, 4) + size(cb, 4))
            core[1:size(ca, 1), :, :, 1:size(ca, 4)] = ca
            core[(size(ca, 1) + 1):end, :, :, (size(ca, 4) + 1):end] = cb
        end
        push!(cores, core)
    end
    return TTop(cores)
end

"Apply the operator to a TT vector and compress the result."
function apply(o::TTop, v::TTvec; cutoff::Float64 = DEFAULT_CUTOFF,
               maxdim::Int = DEFAULT_MAXDIM)
    dims_in(o) == localdims(v) || error("operator input dimensions do not match the vector")
    T = promote_type(eltype(o.cores[1]), eltype(v.cores[1]))
    cores = Array{T,3}[]
    for (oc, vc) in zip(o.cores, v.cores)
        ra, dout, din, rb = size(oc)
        rc, _, re = size(vc)
        # M[a, i, b, c, e] = sum_j oc[a, i, j, b] vc[c, j, e]
        op = reshape(permutedims(oc, (1, 2, 4, 3)), ra * dout * rb, din)
        vp = reshape(permutedims(vc, (2, 1, 3)), din, rc * re)
        m = reshape(op * vp, ra, dout, rb, rc, re)
        # Fuse the left bond as (a, c) with a fastest and the right as
        # (b, e) with b fastest; the same rule at every site keeps the
        # shared bonds of adjacent cores consistent.
        merged = permutedims(m, (1, 4, 2, 3, 5))
        push!(cores, reshape(merged, ra * rc, dout, rb * re))
    end
    return compress(TTvec(cores); cutoff = cutoff, maxdim = maxdim)
end

"Operator product a b, contracted core by core and compressed."
function compose(a::TTop, b::TTop; cutoff::Float64 = DEFAULT_CUTOFF,
                 maxdim::Int = DEFAULT_MAXDIM)
    dims_in(a) == dims_out(b) || error("operator dimensions do not match for composition")
    T = promote_type(eltype(a.cores[1]), eltype(b.cores[1]))
    cores = Array{T,4}[]
    for (ca, cb) in zip(a.cores, b.cores)
        ra, dout, dmid, ra2 = size(ca)
        rb, _, din, rb2 = size(cb)
        # M[a, i, a2, b, k, b2] = sum_j ca[a, i, j, a2] cb[b, j, k, b2]
        ap = reshape(permutedims(ca, (1, 2, 4, 3)), ra * dout * ra2, dmid)
        bp = reshape(permutedims(cb, (2, 1, 3, 4)), dmid, rb * din * rb2)
        m = reshape(ap * bp, ra, dout, ra2, rb, din, rb2)
        merged = permutedims(m, (1, 4, 2, 5, 3, 6))
        push!(cores, reshape(merged, ra * rb, dout, din, ra2 * rb2))
    end
    return compress(TTop(cores); cutoff = cutoff, maxdim = maxdim)
end

"Compressed copy of an operator, reusing the vector compression sweep on the fused physical legs."
function compress(o::TTop; cutoff::Float64 = DEFAULT_CUTOFF, maxdim::Int = DEFAULT_MAXDIM)
    merged = [reshape(c, size(c, 1), size(c, 2) * size(c, 3), size(c, 4)) for c in o.cores]
    slim = compress(TTvec(merged); cutoff = cutoff, maxdim = maxdim)
    cores = [reshape(c, size(c, 1), size(orig, 2), size(orig, 3), size(c, 3))
             for (c, orig) in zip(slim.cores, o.cores)]
    return TTop(cores)
end

"Contract to a dense matrix, first site slowest on both legs; only for small state spaces."
function to_dense(o::TTop{T}) where {T}
    total_out = prod(dims_out(o))
    total_in = prod(dims_in(o))
    total_out * total_in <= 12_000_000 || error("operator too large to densify")
    acc = ones(T, 1, 1, 1)
    for core in o.cores
        r, dout, din, rr = size(core)
        po, pi = size(acc, 1), size(acc, 2)
        # acc[po, pi, r] core[r, i, j, r'] -> new[(po, i), (pi, j), r'] with
        # the new site indices fastest, matching the vector convention.
        m = reshape(acc, po * pi, r) * reshape(core, r, dout * din * rr)
        m = reshape(m, po, pi, dout, din, rr)
        m = permutedims(m, (3, 1, 4, 2, 5))
        acc = reshape(m, dout * po, din * pi, rr)
    end
    return reshape(acc, total_out, total_in)
end

"Sum a list of TT operators, compressing after each addition."
function mpo_sum(operators::Vector{<:TTop}; cutoff::Float64 = DEFAULT_CUTOFF,
                 maxdim::Int = DEFAULT_MAXDIM)
    isempty(operators) && error("mpo_sum needs at least one operator")
    out = operators[1]
    for o in operators[2:end]
        out = compress(add(out, o); cutoff = cutoff, maxdim = maxdim)
    end
    return out
end

"""
Binary-address shift operator on 2^nbits states, without wraparound.

Entry [y, x] is 1 when y = x + step and both lie in 0 .. 2^nbits - 1, for
step in (+1, -1). The construction is the carry chain on the binary digits,
with the first site the most significant bit, matching product_state; the
shift by +1 flips trailing 1s to 0 and the first 0 above them to 1. The
carry that leaves the most significant bit is dropped, which is what
removes wraparound. Bond rank is 2 regardless of nbits. The bond states
are ordered (carry, done).
"""
function qtt_shift(nbits::Int, step::Int)
    nbits >= 1 || error("nbits must be at least 1")
    step in (1, -1) || error("step must be +1 or -1")
    raise_bit = zeros(2, 2)
    raise_bit[2, 1] = 1.0
    lower_bit = zeros(2, 2)
    lower_bit[1, 2] = 1.0
    eye = Matrix{Float64}(I, 2, 2)
    carry, finish = step == 1 ? (lower_bit, raise_bit) : (raise_bit, lower_bit)
    cores = Array{Float64,4}[]
    for k in 1:nbits
        if nbits == 1
            core = zeros(1, 2, 2, 1)
            core[1, :, :, 1] = finish
        elseif k == nbits
            # Least significant site: emits (carry, done) to its left.
            core = zeros(2, 2, 2, 1)
            core[1, :, :, 1] = carry
            core[2, :, :, 1] = finish
        elseif k == 1
            # Most significant site: accepts done only, so an unresolved
            # carry at the top is dropped, removing wraparound.
            core = zeros(1, 2, 2, 2)
            core[1, :, :, 1] = finish
            core[1, :, :, 2] = eye
        else
            core = zeros(2, 2, 2, 2)
            core[1, :, :, 1] = carry
            core[2, :, :, 1] = finish
            core[2, :, :, 2] = eye
        end
        push!(cores, core)
    end
    return TTop(cores)
end
