@testset "tensor-train core against dense linear algebra" begin
    dims = [2, 3, 2, 4]
    total = prod(dims)
    vec_a = randn(total)
    vec_b = randn(total)

    @testset "dense round trip and product-state ordering" begin
        @test isapprox(to_dense(from_dense(vec_a, dims)), vec_a; atol = 1e-12)
        values = [1, 2, 0, 3]
        flat = ((values[1] * 3 + values[2]) * 2 + values[3]) * 4 + values[4]
        dense = to_dense(product_state(dims, values))
        @test dense[flat + 1] == 1.0
        @test sum(dense) == 1.0
    end

    @testset "add, scale, inner, sum, compress" begin
        ta = from_dense(vec_a, dims)
        tb = from_dense(vec_b, dims)
        @test isapprox(to_dense(compress(add(ta, scale(tb, -2.0)))), vec_a - 2 .* vec_b; atol = 1e-10)
        @test isapprox(inner(ta, tb), dot(vec_a, vec_b); atol = 1e-10)
        @test isapprox(sum_entries(ta), sum(vec_a); atol = 1e-10)
        fat = add(ta, tb)
        slim = compress(fat)
        @test maximum(ranks(slim)) <= maximum(ranks(fat))
        @test isapprox(to_dense(slim), vec_a + vec_b; atol = 1e-10)
    end

    @testset "operator apply, transpose, compose, diagonal" begin
        m1 = randn(3, 3)
        m2 = randn(4, 4)
        op = product_operator(dims, Dict(2 => m1, 4 => m2))
        dense = kron(kron(kron(Matrix(I, 2, 2), m1), Matrix(I, 2, 2)), m2)
        @test isapprox(to_dense(op), dense; atol = 1e-12)
        ta = from_dense(vec_a, dims)
        @test isapprox(to_dense(apply(op, ta; cutoff = 0.0)), dense * vec_a; atol = 1e-10)
        @test isapprox(to_dense(tt_transpose(op)), transpose(dense); atol = 1e-12)
        @test isapprox(to_dense(compose(op, op)), dense * dense; atol = 1e-9)
        diag_vec = randn(total)
        @test isapprox(to_dense(diagonal_from_tt(from_dense(diag_vec, dims))),
                       Matrix(Diagonal(diag_vec)); atol = 1e-10)
        total_op = mpo_sum([op, scale(tt_identity(dims), 0.5)])
        @test isapprox(to_dense(total_op), dense + 0.5 * Matrix(I, total, total); atol = 1e-9)
    end

    @testset "binary-address shifts against dense sub and superdiagonals" begin
        for nbits in 1:6
            n = 2^nbits
            up_ref = [Float64(y == x + 1) for y in 0:(n - 1), x in 0:(n - 1)]
            down_ref = [Float64(y == x - 1) for y in 0:(n - 1), x in 0:(n - 1)]
            @test isapprox(to_dense(qtt_shift(nbits, +1)), up_ref; atol = 1e-14)
            @test isapprox(to_dense(qtt_shift(nbits, -1)), down_ref; atol = 1e-14)
        end
        @test ranks(qtt_shift(6, +1)) == vcat([1], fill(2, 5), [1])
    end

    @testset "validation rejects bad input" begin
        @test_throws ErrorException TTvec([zeros(2, 2, 1)])
        @test_throws ErrorException product_state([2, 2], [0, 5])
        @test_throws ErrorException qtt_shift(3, 2)
    end
end
