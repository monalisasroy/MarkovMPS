@testset "solver verbs against dense references, baselines against each other" begin
    @testset "tt_evolve matches the dense exponential with pinned order" begin
        walk = BiasedWalk(3, 1.1, 0.9, 3)
        generator, _, _ = walk_generator(walk)
        dense = to_dense(generator)
        p0 = initial_tt(walk)
        t = 1.5
        reference = exp(dense * t) * to_dense(p0)
        coarse = to_dense(tt_evolve(generator, p0, t, 0.1; order = 4))
        fine = to_dense(tt_evolve(generator, p0, t, 0.05; order = 4))
        # Measured 1.3e-7 at dt = 0.05 on this case in the Python twin;
        # bounded at 1e-6 to hold across linear-algebra backends.
        @test max_abs_deviation(fine, reference) < 1e-6
        # Halving the step must cut the global error by close to 2^4; the
        # measured ratio is 18, bounded below at 10 so the order is pinned.
        @test max_abs_deviation(coarse, reference) / max_abs_deviation(fine, reference) > 10
    end

    @testset "tt_evolve conserves probability" begin
        pile = MannaSandpile(4, 3)
        generator, _, _ = sandpile_generator(pile)
        p = tt_evolve(generator, initial_tt(pile), 1.0, 0.05; order = 4)
        @test abs(real(sum_entries(p)) - 1.0) < 1e-9
    end

    @testset "tt_evolve rejects bad steps" begin
        walk = BiasedWalk(3, 1.0, 1.0, 3)
        generator, _, _ = walk_generator(walk)
        @test_throws ErrorException tt_evolve(generator, initial_tt(walk), -1.0, 0.1)
        @test_throws ErrorException tt_evolve(generator, initial_tt(walk), 1.0, 0.0)
        @test_throws ErrorException tt_evolve(generator, initial_tt(walk), 1.0, 0.1; order = 0)
    end

    @testset "tt_linear_solve against a dense solve" begin
        walk = BiasedWalk(3, 1.2, 0.8, 3)
        generator, _, _ = walk_generator(walk)
        absorb = absorbing_projector_tt(walk)
        A = compress(add(absorb, scale(tt_transpose(generator), -1.0)))
        ones_vec = tt_ones(local_dims(walk))
        b = compress(add(ones_vec, scale(apply(absorb, ones_vec), -1.0)))
        x, residual = tt_linear_solve(A, b; sweeps = 6)
        reference = to_dense(A) \ to_dense(b)
        @test residual < 1e-10
        @test max_abs_deviation(to_dense(x), reference) < 1e-8
    end

    @testset "exact and Monte Carlo agree on the walk" begin
        walk = BiasedWalk(3, 1.2, 0.8, 3)
        mean, err = kmc_mfpt(walk, 3000, 7)
        # 5 standard errors bounds the statistical fluctuation of the sampler.
        @test abs(mean - exact_mfpt(walk)) < 5 * err
        times = [1.0, 3.0]
        @test max_abs_deviation(kmc_survival(walk, times, 2000, 5),
                                exact_survival(walk, times)) < 0.05
    end

    @testset "exact and Monte Carlo agree on the sandpile" begin
        pile = MannaSandpile(5, 4)
        sizes = kmc_avalanche_sizes(pile, 3000, 11)
        dist = exact_avalanche_size_distribution(pile, 2000)
        mean_exact = sum((0:2000) .* dist)
        mean_kmc = sum(sizes) / length(sizes)
        err = sqrt(sum((sizes .- mean_kmc) .^ 2) / (length(sizes) - 1) / length(sizes))
        @test abs(mean_kmc - mean_exact) < 5 * err
    end

    @testset "exact avalanche distribution normalises" begin
        pile = MannaSandpile(4, 3)
        dist = exact_avalanche_size_distribution(pile, 800)
        @test minimum(dist) > -1e-14
        @test abs(sum(dist) - 1.0) < 1e-10
    end

    @testset "dense state limit guard" begin
        pile = MannaSandpile(3, 2)
        W, states, _ = exact_generator(pile)
        @test size(W) == (length(states), length(states))
        # The counter and the enumeration agree where both are affordable,
        # which is what licenses guarding on the counter alone.
        @test state_count(pile) == length(states)
        walk = BiasedWalk(3, 1.0, 1.0, 3)
        @test state_count(walk) == length(enumerate_states(walk))
        # The guard runs before the enumeration, so refusing this model is
        # instant; enumerating its conserved sector (about 1e12 states) was
        # the allocation that once killed the process on the first native run.
        big = MannaSandpile(24, 21)
        @test_throws ErrorException exact_generator(big)
    end
end
