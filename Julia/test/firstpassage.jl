@testset "BiasedWalk: closed forms against dense, tensor train against both" begin
    cases = ((3, 1.0, 1.0, 3), (4, 1.3, 0.7, 5), (4, 0.6, 1.5, 9))

    @testset "closed forms against the dense engine" begin
        for (nbits, up, down, start) in cases
            walk = BiasedWalk(nbits, up, down, start)
            split_dense = exact_absorption_probability(walk, s -> s[1] == walk.last)
            @test abs(split_dense - analytic_splitting(walk)) < 1e-10
            @test abs(exact_mfpt(walk) - analytic_mfpt(walk)) < 1e-9
            times = collect(range(0.0, 6.0; length = 7))
            @test max_abs_deviation(exact_survival(walk, times),
                                    analytic_survival(walk, times)) < 1e-10
        end
    end

    @testset "full transient spectrum against the closed form" begin
        for (nbits, up, down, start) in cases
            walk = BiasedWalk(nbits, up, down, start)
            W, states, _ = exact_generator(walk)
            transient = [i for (i, s) in enumerate(states) if !is_absorbing(walk, s)]
            Q = W[transient, transient]
            observed = sort(real.(eigvals(Q)))
            @test max_abs_deviation(observed, sort(-analytic_decay_rates(walk))) < 1e-10
        end
    end

    @testset "tt_survival against the closed spectral form" begin
        walk = BiasedWalk(4, 1.3, 0.7, 5)
        times = [0.5, 1.0, 2.0, 4.0]
        observed = tt_survival(walk, times; maxdim = 64)
        # Bound: the truncated-Taylor global step error at dt = 0.1;
        # measured 1.6e-8 on this case in the Python twin, bounded at 1e-6
        # to hold across backends.
        @test max_abs_deviation(observed, analytic_survival(walk, times)) < 1e-6
    end

    @testset "tt_mfpt and tt_splitting against the closed forms" begin
        walk = BiasedWalk(4, 1.3, 0.7, 5)
        mfpt, residual = tt_mfpt(walk)
        @test residual < 1e-10
        @test abs(mfpt - analytic_mfpt(walk)) < 1e-8
        split, residual_h = tt_splitting(walk)
        @test residual_h < 1e-10
        @test abs(split - analytic_splitting(walk)) < 1e-8
    end

    @testset "tt_slowest_decay_rate against the closed form" begin
        walk = BiasedWalk(4, 1.3, 0.7, 5)
        @test abs(tt_slowest_decay_rate(walk) - analytic_decay_rates(walk)[1]) < 1e-9
    end

    @testset "unbiased limits" begin
        walk = BiasedWalk(4, 1.0, 1.0, 5)
        @test abs(analytic_splitting(walk) - 5 / walk.last) < 1e-14
        @test abs(analytic_mfpt(walk) - 5 * (walk.last - 5) / 2.0) < 1e-12
    end

    @testset "constructor validation" begin
        @test_throws ErrorException BiasedWalk(1, 1.0, 1.0, 1)
        @test_throws ErrorException BiasedWalk(3, -1.0, 1.0, 3)
        @test_throws ErrorException BiasedWalk(3, 1.0, 1.0, 0)
        @test_throws ErrorException BiasedWalk(3, 1.0, 1.0, 7)
    end
end
