# Problem instances: the input boundary.
#
# A committed TOML file names a run completely: the model, its parameters,
# the observable, and the engines to compare. Validation happens here, at
# the boundary where data enters, and nowhere else: an unrecognised key is
# rejected with a message that names the offending key and lists the
# allowed set, so a misspelled key errors instead of running with a wrong
# default. The internal functions then trust the loader. The same key set
# governs the Julia and Python implementations, and the committed instances
# live once, under problems/ at the repository root, shared by both trees.

using TOML

const ALLOWED_KINDS = Set([
    "barrier_survival", "barrier_mfpt", "manna_avalanche", "manna_survival",
])

const ALLOWED_FIELDS = Set([
    "name", "kind", "output",
    "nbits", "rate_up", "rate_down", "start",
    "sites", "grains", "nmax", "seed_site",
    "tmax", "tpoints", "smax", "nshots", "seed", "maxdim", "cutoff",
    "engines",
])

const ALLOWED_ENGINES = Set(["exact", "kmc", "tt"])

const DEFAULT_OUTPUT_DIR = "results"

struct ProblemInstance
    name::String
    kind::String
    output::String
    nbits::Int
    rate_up::Float64
    rate_down::Float64
    start::Int
    sites::Int
    grains::Int
    nmax::Int
    seed_site::Int
    tmax::Float64
    tpoints::Int
    smax::Int
    nshots::Int
    seed::Int
    maxdim::Int
    cutoff::Float64
    engines::Vector{String}
end

"Instantiate the model the instance's kind names. Site indices in the TOML contract are 0-based, shared with the Python tree, so the Julia seed site adds 1."
function build_model(instance::ProblemInstance)
    if startswith(instance.kind, "barrier")
        return BiasedWalk(instance.nbits, instance.rate_up, instance.rate_down,
                          instance.start)
    end
    return MannaSandpile(instance.sites, instance.grains;
                         nmax = instance.nmax, seed_site = instance.seed_site + 1)
end

"Load and validate one TOML problem instance. An unrecognised field, kind, engine, or out-of-range value errors with a message naming the offender and the allowed set."
function load_problem(path::AbstractString)
    data = TOML.parsefile(path)
    unknown = sort([String(k) for k in keys(data) if !(String(k) in ALLOWED_FIELDS)])
    isempty(unknown) || error(
        "unrecognised problem field(s): $(join(unknown, ", ")). " *
        "allowed fields: $(join(sort(collect(ALLOWED_FIELDS)), ", "))")
    kind = String(get(data, "kind", "barrier_survival"))
    kind in ALLOWED_KINDS || error(
        "unknown kind: $kind. allowed kinds: $(join(sort(collect(ALLOWED_KINDS)), ", "))")
    engines_raw = get(data, "engines", sort(collect(ALLOWED_ENGINES)))
    engines = engines_raw isa String ? [engines_raw] : String.(engines_raw)
    bad = sort([e for e in engines if !(e in ALLOWED_ENGINES)])
    isempty(bad) || error(
        "unknown engine(s): $(join(bad, ", ")). " *
        "allowed engines: $(join(sort(collect(ALLOWED_ENGINES)), ", "))")
    nbits = Int(get(data, "nbits", 4))
    sites = Int(get(data, "sites", 6))
    grains = Int(get(data, "grains", 4))
    instance = ProblemInstance(
        String(get(data, "name", splitext(basename(path))[1])),
        kind,
        String(get(data, "output", DEFAULT_OUTPUT_DIR)),
        nbits,
        Float64(get(data, "rate_up", 1.0)),
        Float64(get(data, "rate_down", 1.0)),
        Int(get(data, "start", 2^(nbits - 1))),
        sites,
        grains,
        Int(get(data, "nmax", grains)),
        Int(get(data, "seed_site", div(sites, 2))),
        Float64(get(data, "tmax", 10.0)),
        Int(get(data, "tpoints", 21)),
        Int(get(data, "smax", 60)),
        Int(get(data, "nshots", 2000)),
        Int(get(data, "seed", 1)),
        Int(get(data, "maxdim", 64)),
        Float64(get(data, "cutoff", 1e-10)),
        engines,
    )
    instance.tmax > 0 || error("tmax must be positive")
    instance.tpoints >= 2 || error("tpoints must be at least 2")
    instance.smax >= 1 || error("smax must be at least 1")
    instance.nshots >= 2 || error("nshots must be at least 2")
    instance.maxdim >= 1 || error("maxdim must be at least 1")
    0 <= instance.cutoff < 1 || error("cutoff must lie in [0, 1)")
    build_model(instance)
    return instance
end

"Resolve named problem instances under problems/; the name all expands to every TOML file."
function problem_paths(names, directory::AbstractString)
    paths = String[]
    for name in names
        if name == "all"
            for f in sort(readdir(directory))
                endswith(f, ".toml") && push!(paths, joinpath(directory, f))
            end
        else
            p = endswith(name, ".toml") ? name : joinpath(directory, name * ".toml")
            isfile(p) || error("problem instance not found: $p")
            push!(paths, p)
        end
    end
    return paths
end
