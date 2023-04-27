# Check that the jacobian is correct
using Pkg
Pkg.activate(".")
Pkg.add(url = "https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl", rev = "main")
using ADNLPModels

using Dates, DelimitedFiles, JLD2, LinearAlgebra, Printf, SparseArrays, Test
# using Pkg.Artifacts
using BenchmarkTools, DataFrames, JuMP, Plots
#JSO packages
using NLPModels, BenchmarkProfiles, NLPModelsJuMP, OptimizationProblems, SolverBenchmark
#This package
using ReverseDiff, Zygote, ForwardDiff

meta = OptimizationProblems.meta
all_problems = meta[meta.ncon .> 0, :name]
for problem in ["arglina"] # all_problems
  problem == "hs61" && continue
  @info "$problem"
  @time nlp_ad = OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))()
  @time nlp_ju = MathOptNLPModel(OptimizationProblems.PureJuMP.eval(Meta.parse(problem))())
  x = get_x0(nlp_ad)
  @test x == get_x0(nlp_ju)
  @time Jx = jac(nlp_ad, x)
  @time Jy = jac(nlp_ju, x)
  @show hess(nlp_ad, x)
  @show hess(nlp_ju, x)
  @test (norm(Jx - Jy) ≈ 0) atol = eps() rtol = norm(Jx)
end
