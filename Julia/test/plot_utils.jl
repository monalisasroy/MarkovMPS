@testset "writer layer: every kind writes both files, tags never collide" begin
    problems_dir = joinpath(@__DIR__, "..", "..", "problems")

    @testset "every committed fast kind writes a table and a figure" begin
        for name in ("barrier", "barrier_mfpt", "manna_small")
            full = load_problem(joinpath(problems_dir, name * ".toml"))
            # Shrink the sampling so the test stays fast; the committed
            # instance keeps its own parameters for the reproducible run.
            small = ProblemInstance(full.name, full.kind, full.output,
                                    full.nbits, full.rate_up, full.rate_down,
                                    full.start, full.sites, full.grains,
                                    full.nmax, full.seed_site, full.tmax, 5,
                                    min(full.smax, 12), 50, full.seed,
                                    full.maxdim, full.cutoff, full.engines)
            outdir = mktempdir()
            dest = run_problem(small; outdir = outdir)
            @test isfile(dest.table) && filesize(dest.table) > 0
            @test isfile(dest.figure) && filesize(dest.figure) > 0
            @test dirname(dest.figure) == joinpath(outdir, "figures")
        end
    end

    @testset "flag tags never collide" begin
        tags = [flag_tag("manna_avalanche", 4, 1.0, 1.0, 8, sites, 4, 10.0, 21, 60, 2000, 1)
                for sites in (4, 5, 6)]
        @test length(unique(tags)) == length(tags)
        # A kind that reads only some fields must still change tag when any
        # field it reads changes, or a sweep silently overwrites output.
        tags_seed = [flag_tag("barrier_mfpt", 4, 1.0, 1.0, 8, 6, 4, 10.0, 21, 60, 2000, seed)
                     for seed in (1, 2, 3)]
        @test length(unique(tags_seed)) == length(tags_seed)
        @test_throws ErrorException flag_tag("barrier_speed", 4, 1.0, 1.0, 8, 6, 4,
                                             10.0, 21, 60, 2000, 1)
    end
end
