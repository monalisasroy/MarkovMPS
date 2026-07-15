#!/usr/bin/env julia
# Headline first-passage run: the committed barrier instances. Runs the
# survival curve and the mean first-passage time of the biased walk with
# all three engines and writes the tables and figures under results/. The
# survival curve lands on the closed spectral form and the hitting time on
# the gambler's ruin value. Run from the Julia directory:
#   julia --project=. examples/barrier_crossing.jl

using MarkovMPS

root = joinpath(@__DIR__, "..", "..")
for name in ("barrier", "barrier_mfpt")
    instance = load_problem(joinpath(root, "problems", name * ".toml"))
    dest = run_problem(instance; outdir = joinpath(root, "results", "julia"))
    @info "wrote" table = dest.table figure = dest.figure
end
