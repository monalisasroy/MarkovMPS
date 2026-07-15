# Test suite includer, mirroring src/: one test file per source module. The
# same closed forms, twin checks, and tolerances are asserted by the Python
# tree's pytest suite.

using LinearAlgebra
using Test
using MarkovMPS

include("tensortrain.jl")
include("generator.jl")
include("solvers.jl")
include("firstpassage.jl")
include("manna.jl")
include("utils.jl")
include("plot_utils.jl")
