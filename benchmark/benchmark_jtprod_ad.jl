#=
In this script, we benchmark several AD-backend.

TODO:
- automate "prepare benchmark" step for more functions
- analyze result
=#
using Pkg
Pkg.activate(".")
Pkg.add(url = "https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl", rev = "main")
using ADNLPModels

using Dates, DelimitedFiles, JLD2, LinearAlgebra, Printf, SparseArrays
# using Pkg.Artifacts
using BenchmarkTools, DataFrames, JuMP, Plots
#JSO packages
using NLPModels, BenchmarkProfiles, NLPModelsJuMP, OptimizationProblems, SolverBenchmark
#This package
using ReverseDiff, Zygote, ForwardDiff

include("utils.jl")

using SparseDiffTools, ReverseDiff, ForwardDiff

struct OptimizedReverseDiffADJtprod{T, S, GT} <: ADNLPModels.InPlaceADbackend
  #ψ::F2
  #gcfg::GC  # gradient config
  gtape::GT  # compiled gradient tape
  _tmp_out::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  _rval::S  # temporary storage for jtprod
end

function OptimizedReverseDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  y0::AbstractVector{T} = rand(ncon),
  kwargs...,
) where {T}
  _tmp_out = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, ncon)
  _rval = similar(x0, ncon)  # temporary storage for jtprod

  ψ(x, u; tmp_out = _tmp_out) = begin
    # here x is a vector of ReverseDiff.TrackedReal
    c!(tmp_out, x)
    dot(tmp_out, u)
  end

  # u = similar(x0, nequ)  # just for GradientConfig
  gcfg = ReverseDiff.GradientConfig((x0, y0))
  gtape = ReverseDiff.compile(ReverseDiff.GradientTape(ψ, (x0, y0), gcfg))

  return OptimizedReverseDiffADJtprod(gtape, _tmp_out, _rval)
end

function ADNLPModels.Jtprod!(b::OptimizedReverseDiffADJtprod, Jtv, c!, x, v)
  ReverseDiff.gradient!((Jtv, b._rval), b.gtape, (x, v))
  return Jtv
end

benchmarked_optimized_backends["jtprod_backend"] = Dict(
  "reverse" => OptimizedReverseDiffADJtprod,
)

########################################################
# There are 6 levels:
# - bench-type (see `benchs`);
# - problem set (see `keys(problem_sets)`);
# - backend name (see `values(tested_backs)`);
# - backend (see `set_back_list(Val(f), test_back)`)
problem_sets = Dict(
  #"all" => setdiff(all_cons_problems, ["camshape"]), # crash
  "scalable" => setdiff(scalable_cons_problems, ["camshape"]),
)
benchs = [
  "optimized",
  #"generic",
]
data_types = [Float64] # [Float16, Float32, Float64]
tested_backs = Dict(
  "jtprod_backend" => :jtprod,
)
const nscal = nn * 100
name = "$(today())_adnlpmodels_benchmark_jtprod"
if "all" in keys(problem_sets)
  name *= "_all"
end
if "generic" in benchs
  name *= "_generic"
elseif "optimized" in benchs
  name *= "_optimized"
end
if "scalable" in keys(problem_sets)
  name *= "_nscal_$(nscal)"
end
if data_types == [Float64]
  name *= "_mono"
else
  name *= "_multi"
end
########################################################
#=
cam shape fails for ReverseDiff
=#

@info "Initialize benchmark"
const SUITE = BenchmarkGroup()

for f in benchs
  SUITE[f] = BenchmarkGroup()
  for s in keys(problem_sets)
    SUITE[f][s] = BenchmarkGroup()
    for test_back in keys(tested_backs)
      back_name = tested_backs[test_back]
      SUITE[f][s][back_name] = BenchmarkGroup()
      for backend in set_back_list(Val(Symbol(f)), test_back)
        SUITE[f][s][back_name][backend] = BenchmarkGroup()
        for T in data_types
          if is_jump_available(Val(Symbol(backend)), T)
            SUITE[f][s][back_name][backend][T] = BenchmarkGroup()
            problems = problem_sets[s]
            for pb in problems
              SUITE[f][s][back_name][backend][T][pb] = BenchmarkGroup()
            end
          end
        end
      end
    end
  end
end

@info "Prepare benchmark"
for f in benchs
  for s in keys(problem_sets)
    for test_back in keys(tested_backs)
      back_name = tested_backs[test_back]
      back_list = set_back_list(Val(Symbol(f)), test_back)
      for backend in back_list
        problems = problem_sets[s]
        for T in data_types
          @info "Prepare $backend with T=$T"
          if !(backend == "jump" && T != Float64)
            for pb in problems
              # add some asserts to make sure it is ok
              n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
              m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
              @info " $(pb): $T with $n vars and $m cons"
              v = [sin(T(i) / 10) for i=1:m]
              SUITE[f][s][back_name][backend][T][pb] = @benchmarkable jtprod(nlp, get_x0(nlp), $v) setup=(nlp = set_problem($pb, $(test_back), $backend, $(f), $s, $nscal, $T))
            end
          end
        end
      end
    end
  end
end

@info "Starting evaluating the benchmark"
result = run(SUITE)

@save "$name.jld2" result
