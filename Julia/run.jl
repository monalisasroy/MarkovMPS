#!/usr/bin/env julia
# CLI entry point over the MarkovMPS library. The library holds the logic;
# this file resolves parameters and dispatches. Named problem instances live
# in problems/ at the repository root, and flag-driven runs use the same
# ProblemInstance type.
# Every parameter flag accepts one or more values; a flag given several
# values repeats the run for each combination in one invocation, the same
# sweep convention as run.py in the Python tree and run.jl in
# QuantumWalks.jl.

const USAGE = """
MarkovMPS run.jl: run first-passage and avalanche checks and write tables and figures into results/ at the repository root.

Usage:
  julia --project=. run.jl --problem NAME [NAME ...] [--output-dir PATH]
  julia --project=. run.jl [--kind KIND] [parameter flags] [--output-dir PATH]

Problem instances:
  julia --project=. run.jl --problem barrier
  julia --project=. run.jl --problem barrier_mfpt
  julia --project=. run.jl --problem manna_small
  julia --project=. run.jl --problem manna_critical
  julia --project=. run.jl --problem all

Flag run:
  --kind NAME          barrier_survival | barrier_mfpt | manna_avalanche | manna_survival
                                                                (default barrier_survival)
  --nbits INT [...]    binary sites of the biased walk          (default 4)
  --rate-up F [...]    upward hop rate                          (default 1.0)
  --rate-down F [...]  downward hop rate                        (default 1.0)
  --start INT [...]    starting position                        (default half range)
  --sites INT [...]    Manna ring size                          (default 6)
  --grains INT [...]   Manna grain count                        (default 4)
  --tmax F [...]       longest time on a survival curve         (default 10.0)
  --tpoints INT [...]  times on a survival curve                (default 21)
  --smax INT [...]     largest avalanche size resolved          (default 60)
  --nshots INT [...]   Monte Carlo trajectories                 (default 2000)
  --seed INT [...]     Monte Carlo seed                         (default 1)
  --engines E [...]    subset of exact kmc tt                   (default all 3)
  --output-dir PATH    directory for run files        (default results/julia at the root)
  -h, --help           show this message and exit

Sweep: any parameter flag above takes one or more values; the run repeats
for every combination of the values given, writing one table and one
figure per combination under a tag built from the fields the kind reads,
so no 2 combinations collide on a filename.

  julia --project=. run.jl --kind manna_avalanche --sites 4 5 6 --grains 3

A long sweep can be detached from the shell:

  nohup julia --project=. run.jl --kind manna_avalanche --sites 4 5 6 \\
    --output-dir results/sweep > sweep.log 2>&1 & echo \$! > sweep.pid
  tail -f sweep.log
  kill "\$(cat sweep.pid)" 2>/dev/null
"""

if any(a -> a == "--help" || a == "-h", ARGS)
    print(USAGE)
    exit(0)
end

using MarkovMPS

function parse_args(args)
    opts = Dict{String,Vector{String}}()
    i = 1
    while i <= length(args)
        tok = args[i]
        if startswith(tok, "--")
            name = tok[3:end]
            vals = String[]
            i += 1
            while i <= length(args) && !startswith(args[i], "--")
                push!(vals, args[i])
                i += 1
            end
            opts[name] = isempty(vals) ? ["true"] : vals
        else
            i += 1
        end
    end
    return opts
end

gets(opts, name, default) = haskey(opts, name) ? first(opts[name]) : default

"Every value given for `name`, or a one-element list holding `default` when the flag was not given. The caller's Cartesian product over these lists is the sweep."
getvals(opts, name, default::AbstractString) = haskey(opts, name) ? opts[name] : [default]
getvals_i(opts, name, default::Int) = parse.(Int, getvals(opts, name, string(default)))
getvals_f(opts, name, default::Float64) = parse.(Float64, getvals(opts, name, string(default)))

function main()
    opts = parse_args(ARGS)
    outdir = gets(opts, "output-dir", joinpath(@__DIR__, "..", "results", "julia"))
    if haskey(opts, "problem")
        dir = joinpath(@__DIR__, "..", "problems")
        for path in problem_paths(opts["problem"], dir)
            instance = load_problem(path)
            dest = run_problem(instance; outdir = outdir)
            @info "wrote" problem = instance.name kind = instance.kind table = dest.table figure = dest.figure
        end
        return
    end

    kind = gets(opts, "kind", "barrier_survival")
    engines = haskey(opts, "engines") ? opts["engines"] : ["exact", "kmc", "tt"]
    nbits_vals = getvals_i(opts, "nbits", 4)
    up_vals = getvals_f(opts, "rate-up", 1.0)
    down_vals = getvals_f(opts, "rate-down", 1.0)
    sites_vals = getvals_i(opts, "sites", 6)
    grains_vals = getvals_i(opts, "grains", 4)
    tmax_vals = getvals_f(opts, "tmax", 10.0)
    tpoints_vals = getvals_i(opts, "tpoints", 21)
    smax_vals = getvals_i(opts, "smax", 60)
    nshots_vals = getvals_i(opts, "nshots", 2000)
    seed_vals = getvals_i(opts, "seed", 1)

    for nbits in nbits_vals, up in up_vals, down in down_vals,
        sites in sites_vals, grains in grains_vals, tmax in tmax_vals,
        tpoints in tpoints_vals, smax in smax_vals, nshots in nshots_vals,
        seed in seed_vals

        starts = haskey(opts, "start") ? parse.(Int, opts["start"]) : [2^(nbits - 1)]
        for start in starts
            tag = flag_tag(kind, nbits, up, down, start, sites, grains,
                           tmax, tpoints, smax, nshots, seed)
            instance = ProblemInstance(tag, kind, outdir, nbits, up, down, start,
                                       sites, grains, grains, div(sites, 2),
                                       tmax, tpoints, smax, nshots, seed,
                                       64, 1e-10, engines)
            dest = run_problem(instance; outdir = outdir)
            @info "wrote" problem = instance.name kind = instance.kind table = dest.table figure = dest.figure
        end
    end
end

main()
