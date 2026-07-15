@testset "problem loader: strict keys and the committed instances" begin
    problems_dir = joinpath(@__DIR__, "..", "..", "problems")

    @testset "committed instances load and build" begin
        for f in sort(readdir(problems_dir))
            endswith(f, ".toml") || continue
            instance = load_problem(joinpath(problems_dir, f))
            model = build_model(instance)
            @test model isa BiasedWalk || model isa MannaSandpile
        end
    end

    @testset "unknown field is rejected" begin
        path = joinpath(mktempdir(), "bad.toml")
        write(path, "name = \"bad\"\nkind = \"barrier_survival\"\nnbitss = 4\n")
        @test_throws ErrorException load_problem(path)
        try
            load_problem(path)
        catch err
            @test occursin("nbitss", err.msg)
        end
    end

    @testset "unknown kind is rejected" begin
        path = joinpath(mktempdir(), "bad.toml")
        write(path, "name = \"bad\"\nkind = \"barrier_speed\"\n")
        try
            load_problem(path)
            @test false
        catch err
            @test occursin("barrier_speed", err.msg)
        end
    end

    @testset "unknown engine is rejected" begin
        path = joinpath(mktempdir(), "bad.toml")
        write(path, "kind = \"barrier_mfpt\"\nengines = [\"exact\", \"dmrg\"]\n")
        try
            load_problem(path)
            @test false
        catch err
            @test occursin("dmrg", err.msg)
        end
    end

    @testset "out-of-range values are rejected" begin
        path = joinpath(mktempdir(), "bad.toml")
        write(path, "kind = \"barrier_survival\"\ntmax = -1.0\n")
        try
            load_problem(path)
            @test false
        catch err
            @test occursin("tmax", err.msg)
        end
    end

    @testset "problem_paths resolves all and rejects a missing name" begin
        @test length(problem_paths(["all"], problems_dir)) >= 4
        @test_throws ErrorException problem_paths(["nonexistent"], problems_dir)
    end
end
