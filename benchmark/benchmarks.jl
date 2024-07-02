# Include useful packages
using ADNLPModels
using Dates, DelimitedFiles, JLD2, LinearAlgebra, Printf, SparseArrays
using BenchmarkTools, DataFrames
#JSO packages
using NLPModels, OptimizationProblems, SolverBenchmark
# Most likely benchmark with JuMP as well
using JuMP, NLPModelsJuMP

include("problems_sets.jl")
verbose_subbenchmark = false

# Run locally with `tune!(SUITE)` and then `run(SUITE)`
const SUITE = BenchmarkGroup()

include("gradient/benchmarks_gradient.jl")

include("jacobian/benchmarks_coloring.jl")
include("jacobian/benchmarks_jacobian.jl")
include("jacobian/benchmarks_jacobian_residual.jl")

include("hessian/benchmarks_coloring.jl")
include("hessian/benchmarks_hessian.jl")
include("hessian/benchmarks_hessian_lagrangian.jl")
include("hessian/benchmarks_hessian_residual.jl")

include("jacobian/benchmarks_jprod.jl")
include("jacobian/benchmarks_jprod_residual.jl")
include("jacobian/benchmarks_jtprod.jl")
include("jacobian/benchmarks_jtprod_residual.jl")
