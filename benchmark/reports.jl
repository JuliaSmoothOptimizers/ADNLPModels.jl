using Pkg; Pkg.activate(".")
Pkg.add(url = "https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl", rev = "main")
using ADNLPModels
#This package
using ForwardDiff, ReverseDiff, SparseDiffTools, Zygote

using Dates, DelimitedFiles, JLD2, LinearAlgebra, Printf, SparseArrays
using BenchmarkTools, DataFrames, Plots # JuMP
# JSO
using NLPModels, BenchmarkProfiles, OptimizationProblems

include("additional_backends.jl")
include("utils.jl")

# Some problem sets are prepared
pre_problem_sets

# They contain list of names
pre_problem_sets["all"]

# We can also access the usual metadata
meta = OptimizationProblems.meta;
meta[
    map(x -> x in pre_problem_sets["all"], meta.name), 
    [:name, :nvar, :ncon]
]

# The backend are divided in two
# - optimized: that typically run pre-computations
# - generic: that do not.
# The file additional_backends.jl contains a list of backend not (yet) in ADNLPModels.
# The available backend are grouped in dict:
benchmarked_optimized_backends

# and the generic ones
benchmarked_generic_backends

# the benchmark files will also add jump for the comparison
# if the type is Float64.
# The result of the benchmark is stored in a jld2 file with the name
# `date`_adnlpmodels_benchmark_`operation`_`set`_`type`.jld2
# type: is either `mono` (if only Float64) or `mutli`;
# set: `all` or `nscal_100`
# operation: grad, jprod, jtprod, hprod, jac ...
# The file figures.jl print a simple perf profile.

# TODO:
# - parallelized the benchmark
# - track the evolution of the benchmark

name = "2023-03-23_adnlpmodels_benchmark_hprod_optimized_nscal_1000_mono"
@load "$name.jld2" result
result

# Some operations have no optimized backends:
# - hprod: ongoing PR ForwardDiff/SparseDiffTools
# - jtprod: ongoing PR ReverseDiff
# - hessian: work with Alexis

# Some more to be explored (later):
# - Nabla (Invenia)
# - Enzyme (Argonne)
# Others from JuliaDiff
# - https://github.com/JuliaDiff/PolyesterForwardDiff.jl
# - https://github.com/JuliaDiff/Diffractor.jl 

# Conclusion:
# We need larger scale benchmark, but on n = 1000 we are doing overall OK.

# New features of ADNLPModels:
# - in-place constraints and residual;
# - should also be updated in OptimizationProblems (for the benchmarks);
# - add documentations for the new features and review.

# Goals:
# 1) finalize the 3 missing backends;
# 2) in-place functions in OptimizationProblems;
# 3) re-do benchmarks (with n=...)
# 4) documentation
# Later:
# test other backends
# Later later:
# mutli-precision.
