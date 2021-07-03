using DelimitedFiles, LinearAlgebra, Printf, SparseArrays
# using Pkg.Artifacts
using BenchmarkTools, DataFrames, JuMP, Plots
#JSO packages
using NLPModels, BenchmarkProfiles, SolverBenchmark, NLPModelsJuMP
#This package
using ADNLPModels, ReverseDiff, Zygote, ForwardDiff

include("problems/problems.jl")

models = [:reverse, :zygote, :autodiff, :jump]
fun    = Dict(:obj => (nlp, x) -> obj(nlp, x), 
              :grad => (nlp, x) -> grad(nlp, x),
              :hess_coord => (nlp, x) -> hess_coord(nlp, x), 
              :hess_structure => (nlp, x) -> hess_structure(nlp),
              :jac_coord => (nlp, x) -> (nlp.meta.ncon > 0 ? jac_coord(nlp, x) : zero(eltype(x))),
              :jac_structure => (nlp, x) -> (nlp.meta.ncon > 0 ? jac_structure(nlp) : zero(eltype(x)))
              :hess_lag_coord => (nlp, x) -> hess_coord(nlp, x, ones(nlp.meta.ncon)),
              )
funsym = keys(fun)

const SUITE = BenchmarkGroup()
for f in keys(fun)
  SUITE[f] = BenchmarkGroup()
  for m in models
    SUITE[f][m] = BenchmarkGroup()
  end
end

for pb in problems, m in models
  npb = eval(Meta.parse("$(pb)_$(m)()"))
  for (fs, f) in fun
    x = npb.meta.x0
    SUITE[fs][m][string(pb)] = @benchmarkable eval($f)($npb, $x)
  end
end
