module MarkovMPS

# Files are included in dependency order, so this module file is the package
# map. tensortrain holds the dependency-free tensor-train core; generator
# assembles the escape-diagonal generator from a model's terms; solvers
# holds the 2 permanent baselines (exact dense, kinetic Monte Carlo) and the
# 3 tensor-train verbs (transient stepper, linear solve, eigensolve);
# models/ holds one problem struct per file, the declared extension surface;
# utils validates TOML problem instances; plot_utils writes reproducible
# outputs, tabular and visual, from a completed run. The Python tree beside
# this package holds the same structure file for file.

using LinearAlgebra

include("tensortrain.jl")
include("generator.jl")
include("solvers.jl")
include("models/firstpassage.jl")
include("models/manna.jl")
include("utils.jl")
include("plot_utils.jl")

export DEFAULT_CUTOFF, DEFAULT_MAXDIM, DENSE_STATE_LIMIT,
    TTvec, TTop, product_state, tt_ones, from_dense, to_dense, scale, add,
    inner, sum_entries, tt_norm, compress, tt_sum, tt_identity,
    product_operator, diagonal_from_tt, tt_transpose, apply, compose,
    mpo_sum, qtt_shift, nsites, localdims, dims_in, dims_out, ranks,
    TermSpec, offdiagonal_operator, escape_rates, markov_generator,
    exact_generator, exact_survival, exact_mfpt, exact_absorption_probability,
    exact_avalanche_size_distribution, exact_avalanche_pgf,
    kmc_trajectory, kmc_survival, kmc_avalanche_sizes, kmc_mfpt,
    tt_evolve, tt_linear_solve, tt_eigen_max, max_abs_deviation,
    BiasedWalk, local_dims, state_to_sites, enumerate_states, state_count, is_absorbing,
    moves, initial_state, initial_tt, transient_projector_tt,
    absorbing_projector_tt, terms, rho, analytic_splitting, analytic_mfpt,
    analytic_decay_rates, analytic_survival, analytic_fpt_density,
    walk_generator, tt_survival, tt_mfpt, tt_splitting, tt_slowest_decay_rate,
    MannaSandpile, activity, activity_tt, grain_sector_projector_tt,
    sandpile_generator, assemble_pgf_parts, tt_avalanche_pgf,
    tt_mean_avalanche_size, tt_avalanche_size_distribution,
    ProblemInstance, load_problem, problem_paths, build_model,
    ALLOWED_KINDS, ALLOWED_FIELDS, ALLOWED_ENGINES,
    PLOT_DEFAULTS, run_tag, flag_tag, write_table, run_problem

end # module MarkovMPS
