@testset "generator assembly: twin check and column-sum-zero invariant" begin
    @testset "walk generator equals dense-from-moves" begin
        for (nbits, up, down) in ((3, 1.0, 1.0), (4, 1.3, 0.7))
            walk = BiasedWalk(nbits, up, down, 1)
            W_tt, _, _ = markov_generator(local_dims(walk), terms(walk))
            dense_tt = to_dense(W_tt)
            dense_ref, _, _ = exact_generator(walk)
            # The tensor-train assembly and the moves() enumeration are
            # independent constructions of the same generator; their
            # agreement checks both.
            @test max_abs_deviation(dense_tt, dense_ref) < 1e-11
            @test maximum(abs.(sum(dense_tt; dims = 1))) < 1e-11
        end
    end

    @testset "Manna generator equals dense-from-moves on the full product space" begin
        pile = MannaSandpile(3, 2)
        W_tt, _, escape = markov_generator(local_dims(pile), terms(pile))
        dense_tt = to_dense(W_tt)
        d = pile.nmax + 1
        all_states = [[a, b, c] for a in 0:(d - 1) for b in 0:(d - 1) for c in 0:(d - 1)]
        # Full-space enumeration puts the first site slowest, matching to_dense.
        all_states = sort(all_states; by = s -> (s[1], s[2], s[3]))
        index = Dict(s => i for (i, s) in enumerate(all_states))
        dense_ref = zeros(length(all_states), length(all_states))
        for s in all_states
            i = index[s]
            for (rate, dest) in moves(pile, s)
                dense_ref[index[dest], i] += rate
                dense_ref[i, i] -= rate
            end
        end
        @test max_abs_deviation(dense_tt, dense_ref) < 1e-10
        @test maximum(abs.(sum(dense_tt; dims = 1))) < 1e-10
        # On the conserved sector the escape rate equals the active-site count.
        act = to_dense(activity_tt(pile))
        esc = to_dense(escape)
        for s in enumerate_states(pile)
            i = index[s]
            @test abs(act[i] - esc[i]) < 1e-10
        end
    end

    @testset "ones annihilated by the generator" begin
        pile = MannaSandpile(4, 3)
        W_tt, _, _ = markov_generator(local_dims(pile), terms(pile))
        @test tt_norm(apply(tt_transpose(W_tt), tt_ones(local_dims(pile)))) < 1e-9
    end

    @testset "bad terms are rejected" begin
        @test_throws ErrorException offdiagonal_operator([2, 2], TermSpec[])
        bad = TermSpec[(-1.0, Dict(1 => Matrix{Float64}(I, 2, 2)))]
        @test_throws ErrorException offdiagonal_operator([2, 2], bad)
    end
end
