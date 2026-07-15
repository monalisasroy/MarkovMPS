# The Markov generator: one assembly, written once.
#
# A continuous-time Markov process is defined by its off-diagonal rates:
# the entry W[y, x] is the rate of the move from configuration x to
# configuration y. Probability conservation then fixes the diagonal: the
# escape rate out of x is the sum of the column above it, so every column
# of the full generator sums to 0 and dp/dt = W p conserves total
# probability. The escape diagonal is the one place a sign error corrupts
# every result, so it is assembled here once, from the off-diagonal part
# itself, and every model reuses this function rather than restating the
# bookkeeping.
#
# The assembly takes a model's terms: each term is a rate together with
# either a mapping from sites to local matrices, which becomes a rank-1
# product operator, or a prebuilt tensor-train operator for moves that are
# not site-local products, such as the carry-chain shifts of the biased
# walk. The column sums are computed by applying the transpose of the
# off-diagonal operator to the all-ones vector, so a move blocked at a
# boundary or an occupation cap is excluded from the escape rate
# automatically, with no per-model boundary bookkeeping.

const TermSpec = Tuple{Float64,Union{Dict{Int,Matrix{Float64}},TTop}}

"Sum the model's transition terms into one off-diagonal operator. Columns of absorbing configurations are 0 because no term moves out of them."
function offdiagonal_operator(dims::Vector{Int}, terms::Vector{<:TermSpec};
                              cutoff::Float64 = DEFAULT_CUTOFF)
    isempty(terms) && error("a model must provide at least one transition term")
    operators = TTop[]
    for (rate, spec) in terms
        rate > 0 || error("transition rates must be positive, got $rate")
        if spec isa TTop
            push!(operators, scale(spec, rate))
        else
            push!(operators, scale(product_operator(dims, spec), rate))
        end
    end
    return mpo_sum(operators; cutoff = cutoff, maxdim = OPERATOR_MAXDIM)
end

"Column sums of the off-diagonal operator: the escape rate at each state. Computed as the transpose applied to the all-ones vector, so a blocked move contributes nothing, matching the model's moves() exactly."
function escape_rates(offdiag::TTop; cutoff::Float64 = DEFAULT_CUTOFF)
    return apply(tt_transpose(offdiag), tt_ones(dims_out(offdiag));
                 cutoff = cutoff, maxdim = OPERATOR_MAXDIM)
end

"""
Assemble the full generator W = T - diag(escape) from a model's terms.

Returns (generator, offdiagonal, escape): the full generator with columns
summing to 0, the off-diagonal part T, and the escape-rate vector,
returned separately because the avalanche generating function tilts T
while keeping the escape diagonal fixed.
"""
function markov_generator(dims::Vector{Int}, terms::Vector{<:TermSpec};
                          cutoff::Float64 = DEFAULT_CUTOFF)
    offdiag = offdiagonal_operator(dims, terms; cutoff = cutoff)
    escape = escape_rates(offdiag; cutoff = cutoff)
    diagonal = diagonal_from_tt(escape)
    generator = compress(add(offdiag, scale(diagonal, -1.0));
                         cutoff = cutoff, maxdim = OPERATOR_MAXDIM)
    return generator, offdiag, escape
end
