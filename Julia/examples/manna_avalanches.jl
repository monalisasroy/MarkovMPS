#!/usr/bin/env julia
# Headline avalanche run: the committed Manna instances. manna_small
# compares all three engines on a 4-site ring, including the machine-precision
# tail of the exact embedded chain and the Fourier-inverted tensor-train
# generating function. manna_critical runs kinetic Monte Carlo alone on a
# 24-site ring near the critical density, the stated boundary of the
# package. Run from the Julia directory:
#   julia --project=. examples/manna_avalanches.jl

using MarkovMPS

root = joinpath(@__DIR__, "..", "..")
for name in ("manna_small", "manna_critical")
    instance = load_problem(joinpath(root, "problems", name * ".toml"))
    dest = run_problem(instance; outdir = joinpath(root, "results", "julia"))
    @info "wrote" table = dest.table figure = dest.figure
end
