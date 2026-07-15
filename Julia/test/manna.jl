@testset "MannaSandpile: sector structure, tensor train against the dense twin" begin
    @testset "seeded configuration is deterministic" begin
        pile = MannaSandpile(6, 4)
        occ = initial_state(pile)
        @test sum(occ) == 4
        @test occ[pile.seed_site] == 2
        @test activity(pile, occ) == 1
        @test initial_state(MannaSandpile(6, 4)) == occ
    end

    @testset "moves conserve grains" begin
        pile = MannaSandpile(5, 4)
        for state in enumerate_states(pile)
            for (rate, dest) in moves(pile, state)
                @test sum(dest) == pile.grains
                @test rate > 0
            end
        end
    end

    @testset "sector enumeration counts" begin
        # Compositions of 3 grains over 4 sites capped at 3: C(3 + 4 - 1, 3) = 20.
        @test length(enumerate_states(MannaSandpile(4, 3))) == 20
    end

    @testset "grain sector projector matches enumeration" begin
        pile = MannaSandpile(4, 3)
        projector = to_dense(grain_sector_projector_tt(pile))
        d = pile.nmax + 1
        all_states = [[a, b, c, e] for a in 0:(d - 1) for b in 0:(d - 1)
                      for c in 0:(d - 1) for e in 0:(d - 1)]
        reference = Matrix(Diagonal([sum(s) == pile.grains ? 1.0 : 0.0 for s in all_states]))
        @test isapprox(projector, reference; atol = 1e-12)
    end

    @testset "tt_survival against the exact engine" begin
        pile = MannaSandpile(4, 3)
        times = [0.5, 1.5, 3.0]
        observed = tt_survival(pile, times; maxdim = 48)
        reference = exact_survival(pile, times)
        # Bound: truncated-Taylor step error; measured 1.2e-9 at N = 5 in
        # the Python twin, bounded at 1e-6 to hold across backends.
        @test max_abs_deviation(observed, reference) < 1e-6
    end

    @testset "tt generating function against the dense twin" begin
        pile = MannaSandpile(4, 3)
        for z in (0.5, 0.9, exp(0.4im))
            value, residual, _ = tt_avalanche_pgf(pile, z; maxdim = 48)
            @test residual < 1e-10
            @test abs(value - exact_avalanche_pgf(pile, z)) < 1e-9
        end
    end

    @testset "tt mean avalanche size against the exact engine" begin
        pile = MannaSandpile(4, 3)
        mean_tt, residual = tt_mean_avalanche_size(pile; maxdim = 48)
        @test residual < 1e-10
        dist = exact_avalanche_size_distribution(pile, 800)
        @test abs(mean_tt - sum((0:800) .* dist)) < 1e-8
    end

    @testset "tt size distribution against the exact engine" begin
        pile = MannaSandpile(4, 3)
        smax = 20
        observed = tt_avalanche_size_distribution(pile, smax; maxdim = 48, sweeps = 6)
        reference = exact_avalanche_size_distribution(pile, smax)
        tail = 1.0 - sum(exact_avalanche_size_distribution(pile, 2 * smax + 1))
        # The Fourier inversion aliases the tail beyond 2 smax + 2 points,
        # so the deviation is bounded by that tail mass plus solver tolerance.
        @test max_abs_deviation(observed, reference) < tail + 1e-7
    end

    @testset "constructor validation" begin
        @test_throws ErrorException MannaSandpile(2, 2)
        @test_throws ErrorException MannaSandpile(4, 1)
        @test_throws ErrorException MannaSandpile(4, 5)
        @test_throws ErrorException MannaSandpile(4, 3; nmax = 1)
        @test_throws ErrorException MannaSandpile(4, 3; seed_site = 9)
    end
    @testset "solution cap does not corrupt the operator" begin
        # Regression: the assembly once compressed the operator itself under
        # the solution's maxdim; at maxdim 8, below the middle-bond rank 15
        # of this operator, the solve satisfied a corrupted generator to
        # machine precision while the mean was wrong by half its value.
        # Operators are now compressed by the cutoff alone, so a small
        # solution cap must either reproduce the exact mean or report a
        # residual on the scale of its error.
        pile = MannaSandpile(4, 3)
        dist = exact_avalanche_size_distribution(pile, 800)
        mean_exact = sum((0:800) .* dist)
        for maxdim in (4, 8)
            mean_tt, residual = tt_mean_avalanche_size(pile; maxdim = maxdim)
            error = abs(mean_tt - mean_exact)
            @test !(residual < 1e-8 && error > 1e-6)
        end
        mean_tt, _ = tt_mean_avalanche_size(pile; maxdim = 8)
        @test abs(mean_tt - mean_exact) < 1e-9
    end
end
