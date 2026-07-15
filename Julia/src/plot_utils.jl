# Writing and plotting layer. Source returns values; this file is the only
# place that creates runtime output files, tabular (.tsv) and visual (.png).
# The 2 live in one file because both are peripheral to the physics:
# neither computes a result, each only records or draws one already
# computed and tested elsewhere. Every problem kind writes one table into
# the output directory and one matching figure under figures/, both under
# the same base name, using only values the production path already
# computes, so a figure carries no numerical claim beyond what is already
# trusted.

using Plots

"Default styling keyword arguments for every plot this module produces."
const PLOT_DEFAULTS = (
    size = (760, 480),
    dpi = 200,
    guidefontsize = 12,
    tickfontsize = 10,
    legendfontsize = 9,
    titlefontsize = 12,
    linewidth = 2,
    framestyle = :box,
    background_color = :white,
)

# Colour encodes the role of a quantity, not the problem kind, so the same
# colour means the same thing in every figure.
const COLOR_CLOSED = RGB(0.10, 0.10, 0.10)   # near black: a closed-form anchor
const COLOR_EXACT = RGB(0.13, 0.40, 0.67)    # blue: the dense engine
const COLOR_KMC = RGB(0.70, 0.09, 0.17)      # red: kinetic Monte Carlo
const COLOR_TT = RGB(0.11, 0.47, 0.22)       # green: the tensor train
const COLOR_GUIDE = RGB(0.53, 0.53, 0.53)    # grey: an exponent guide line

"Base name shared by the table and the figure of one run."
run_tag(instance::ProblemInstance) = "$(instance.kind)_$(instance.name)"

"""
Deterministic output tag for one flag-driven parameter combination.

The tag carries only the fields the given kind reads, so 2 combinations
that differ in any one of them never collide on the output filename. Kept
in the library, not the entry file, so the tag rule is unit-tested
directly.
"""
function flag_tag(kind::AbstractString, nbits::Int, rate_up::Float64,
                  rate_down::Float64, start::Int, sites::Int, grains::Int,
                  tmax::Float64, tpoints::Int, smax::Int, nshots::Int, seed::Int)
    if kind == "barrier_survival"
        return "flag_nbits$(nbits)_up$(rate_up)_down$(rate_down)_start$(start)_tmax$(tmax)_tpoints$(tpoints)_nshots$(nshots)_seed$(seed)"
    elseif kind == "barrier_mfpt"
        return "flag_nbits$(nbits)_up$(rate_up)_down$(rate_down)_start$(start)_nshots$(nshots)_seed$(seed)"
    elseif kind == "manna_avalanche"
        return "flag_sites$(sites)_grains$(grains)_smax$(smax)_nshots$(nshots)_seed$(seed)"
    elseif kind == "manna_survival"
        return "flag_sites$(sites)_grains$(grains)_tmax$(tmax)_tpoints$(tpoints)_nshots$(nshots)_seed$(seed)"
    else
        error("unknown kind: $kind")
    end
end

"Write named columns of equal length as one tab-separated table with a header line."
function write_table(path::AbstractString, header::Vector{String},
                     columns::Vector{<:AbstractVector})
    length(header) == length(columns) || error("header and column counts differ")
    open(path, "w") do io
        println(io, join(header, "\t"))
        for i in 1:length(columns[1])
            println(io, join([string(col[i]) for col in columns], "\t"))
        end
    end
    return path
end

function ensure_results_tree(outdir::AbstractString)
    mkpath(outdir)
    mkpath(joinpath(outdir, "figures"))
    return outdir
end

"""
Run one problem instance: compute, write the table, draw the figure.

Returns a named tuple (table, figure) with the 2 files the run produced.
"""
function run_problem(instance::ProblemInstance; outdir::Union{AbstractString,Nothing} = nothing)
    directory = outdir === nothing ? instance.output : outdir
    ensure_results_tree(directory)
    tag = run_tag(instance)
    table_path = joinpath(directory, tag * ".tsv")
    figure_path = joinpath(directory, "figures", tag * ".png")
    if instance.kind == "barrier_survival"
        run_barrier_survival(instance, table_path, figure_path)
    elseif instance.kind == "barrier_mfpt"
        run_barrier_mfpt(instance, table_path, figure_path)
    elseif instance.kind == "manna_avalanche"
        run_manna_avalanche(instance, table_path, figure_path)
    elseif instance.kind == "manna_survival"
        run_manna_survival(instance, table_path, figure_path)
    else
        error("unknown kind: $(instance.kind)")
    end
    return (table = table_path, figure = figure_path)
end

function run_barrier_survival(instance::ProblemInstance, table_path, figure_path)
    walk = build_model(instance)
    times = collect(range(0.0, instance.tmax; length = instance.tpoints))
    closed = analytic_survival(walk, times)
    header = ["time", "closed_form"]
    columns = AbstractVector[times, closed]
    series = Dict{String,Vector{Float64}}()
    if "exact" in instance.engines
        series["exact"] = exact_survival(walk, times)
    end
    if "kmc" in instance.engines
        series["kmc"] = kmc_survival(walk, times, instance.nshots, instance.seed)
    end
    if "tt" in instance.engines
        series["tt"] = tt_survival(walk, times; cutoff = instance.cutoff,
                                   maxdim = instance.maxdim)
    end
    for name in ("exact", "kmc", "tt")
        haskey(series, name) || continue
        push!(header, name)
        push!(columns, series[name])
    end
    write_table(table_path, header, columns)

    title = haskey(series, "tt") ?
        "survival, largest tensor-train deviation $(round(max_abs_deviation(series["tt"], closed); sigdigits = 2))" :
        "survival"
    plt = plot(times, closed; seriescolor = COLOR_CLOSED, label = "closed form",
               xlabel = "time", ylabel = "survival probability", yscale = :log10,
               title = title, legend = :bottomleft, PLOT_DEFAULTS...)
    markers = Dict("exact" => (:circle, COLOR_EXACT), "kmc" => (:square, COLOR_KMC),
                   "tt" => (:utriangle, COLOR_TT))
    for name in ("exact", "kmc", "tt")
        haskey(series, name) || continue
        shape, colour = markers[name]
        scatter!(plt, times, series[name]; markershape = shape, markersize = 4,
                 markerstrokewidth = 1, markercolor = :white,
                 markerstrokecolor = colour, label = name)
    end
    savefig(plt, figure_path)
end

function run_barrier_mfpt(instance::ProblemInstance, table_path, figure_path)
    walk = build_model(instance)
    closed = analytic_mfpt(walk)
    names = String[]
    values = Float64[]
    errors = Float64[]
    if "exact" in instance.engines
        push!(names, "exact"); push!(values, exact_mfpt(walk)); push!(errors, 0.0)
    end
    if "kmc" in instance.engines
        mean, err = kmc_mfpt(walk, instance.nshots, instance.seed)
        push!(names, "kmc"); push!(values, mean); push!(errors, err)
    end
    if "tt" in instance.engines
        mfpt, residual = tt_mfpt(walk; cutoff = instance.cutoff, maxdim = instance.maxdim)
        push!(names, "tt"); push!(values, mfpt); push!(errors, residual)
    end
    write_table(table_path, ["engine", "mfpt", "error_or_residual"],
                AbstractVector[vcat(["closed_form"], names),
                               vcat([closed], values),
                               vcat([0.0], errors)])

    palette = Dict("exact" => COLOR_EXACT, "kmc" => COLOR_KMC, "tt" => COLOR_TT)
    plt = bar(names, values; color = [palette[n] for n in names], label = "",
              ylabel = "mean first-passage time", PLOT_DEFAULTS...)
    hline!(plt, [closed]; linestyle = :dash, seriescolor = COLOR_CLOSED,
           label = "closed form $(round(closed; digits = 4))")
    savefig(plt, figure_path)
end

function run_manna_avalanche(instance::ProblemInstance, table_path, figure_path)
    pile = build_model(instance)
    smax = instance.smax
    sizes = collect(0:smax)
    header = ["size"]
    columns = AbstractVector[sizes]
    series = Dict{String,Vector{Float64}}()
    if "exact" in instance.engines
        series["exact"] = exact_avalanche_size_distribution(pile, smax)
    end
    if "kmc" in instance.engines
        samples = kmc_avalanche_sizes(pile, instance.nshots, instance.seed)
        histogram = zeros(smax + 1)
        for s in samples
            s <= smax && (histogram[s + 1] += 1)
        end
        series["kmc"] = histogram ./ length(samples)
    end
    if "tt" in instance.engines
        series["tt"] = tt_avalanche_size_distribution(pile, smax;
                                                      cutoff = instance.cutoff,
                                                      maxdim = instance.maxdim)
    end
    for name in ("exact", "kmc", "tt")
        haskey(series, name) || continue
        push!(header, name)
        push!(columns, series[name])
    end
    write_table(table_path, header, columns)

    plt = plot(; xlabel = "avalanche size (topplings)", ylabel = "probability",
               xscale = :log10, yscale = :log10, legend = :bottomleft,
               PLOT_DEFAULTS...)
    positive = sizes .>= 1
    if haskey(series, "exact")
        mask = positive .& (series["exact"] .> 1e-300)
        plot!(plt, sizes[mask], series["exact"][mask]; seriescolor = COLOR_EXACT,
              label = "exact embedded chain")
    end
    if haskey(series, "kmc")
        mask = positive .& (series["kmc"] .> 0)
        scatter!(plt, sizes[mask], series["kmc"][mask]; markershape = :square,
                 markersize = 4, markerstrokewidth = 1, markercolor = :white,
                 markerstrokecolor = COLOR_KMC,
                 label = "kinetic Monte Carlo, $(instance.nshots) runs")
    end
    if haskey(series, "tt")
        mask = positive .& (series["tt"] .> 1e-14)
        scatter!(plt, sizes[mask], series["tt"][mask]; markershape = :utriangle,
                 markersize = 4, markerstrokewidth = 1, markercolor = :white,
                 markerstrokecolor = COLOR_TT, label = "tensor train")
    end
    anchor = 0.0
    for name in ("exact", "tt", "kmc")
        if haskey(series, name) && series[name][2] > 0
            anchor = series[name][2]
            break
        end
    end
    if anchor > 0
        guide = anchor .* Float64.(sizes[positive]) .^ (-1.5)
        plot!(plt, sizes[positive], guide; linestyle = :dot, seriescolor = COLOR_GUIDE,
              label = "mean-field exponent guide, size^(-3/2)")
    end
    savefig(plt, figure_path)
end

function run_manna_survival(instance::ProblemInstance, table_path, figure_path)
    pile = build_model(instance)
    times = collect(range(0.0, instance.tmax; length = instance.tpoints))
    header = ["time"]
    columns = AbstractVector[times]
    series = Dict{String,Vector{Float64}}()
    if "exact" in instance.engines
        series["exact"] = exact_survival(pile, times)
    end
    if "kmc" in instance.engines
        series["kmc"] = kmc_survival(pile, times, instance.nshots, instance.seed)
    end
    if "tt" in instance.engines
        series["tt"] = tt_survival(pile, times; cutoff = instance.cutoff,
                                   maxdim = instance.maxdim)
    end
    for name in ("exact", "kmc", "tt")
        haskey(series, name) || continue
        push!(header, name)
        push!(columns, series[name])
    end
    write_table(table_path, header, columns)

    plt = plot(; xlabel = "time", ylabel = "probability the pile is still active",
               yscale = :log10, legend = :bottomleft, PLOT_DEFAULTS...)
    styles = Dict("exact" => COLOR_EXACT, "kmc" => COLOR_KMC, "tt" => COLOR_TT)
    for name in ("exact", "kmc", "tt")
        haskey(series, name) || continue
        plot!(plt, times, series[name]; seriescolor = styles[name], label = name)
    end
    savefig(plt, figure_path)
end
