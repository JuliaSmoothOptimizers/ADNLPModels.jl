using DelimitedFiles, LinearAlgebra, Printf, SparseArrays
# using Pkg.Artifacts
using BenchmarkTools, DataFrames, JuMP, Plots
#JSO packages
using NLPModels, BenchmarkProfiles, SolverBenchmark, NLPModelsJuMP
#This package
using ADNLPModels, ADNLPModelProblems, ReverseDiff, Zygote, ForwardDiff

# Scalable problems from ADNLPModelProblems.jl
const problems =
  ["clnlbeam", "controlinvestment", "hovercraft1d", "polygon1", "polygon2", "polygon3"]

nn = ADNLPModelProblems.default_nvar # 100 # default parameter for scalable problems
# available functions:
# $(pb)_autodiff(args... ; n=$(nn), kwargs...)
# $(pb)_reverse(args... ; kwargs...)
# $(pb)_zygote(args... ; kwargs...)
# $(pb)_jump(args... ; n=$(nn), kwargs...)

models = [:reverse, :zygote, :autodiff, :jump]
fun = Dict(
  :obj => (nlp, x) -> obj(nlp, x),
  :grad => (nlp, x) -> grad(nlp, x),
  :hess_coord => (nlp, x) -> hess_coord(nlp, x),
  :hess_structure => (nlp, x) -> hess_structure(nlp),
  :jac_coord => (nlp, x) -> (nlp.meta.ncon > 0 ? jac_coord(nlp, x) : zero(eltype(x))),
  :jac_structure => (nlp, x) -> (nlp.meta.ncon > 0 ? jac_structure(nlp) : zero(eltype(x))),
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
  npb = eval(Meta.parse("ADNLPModelProblems.$(pb)_$(m)()")) # we should add a kwargs n=(size_of_problem) to modify the size
  for (fs, f) in fun
    x = npb.meta.x0
    SUITE[fs][m][string(pb)] = @benchmarkable eval($f)($npb, $x)
  end
end
