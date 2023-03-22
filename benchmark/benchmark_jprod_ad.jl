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

#=
struct ForwardDiffAD{T, F} <: ADBackend where {T, F <: Function}
  r!::F
  tmp_in::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  tmp_out::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
end

function ForwardDiffAD(r!::F, T::DataType, nvar::Int, nequ::Int) where {F <: Function}
  tmp_in = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  tmp_out = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nequ)
  ForwardDiffAD{T, F}(r!, tmp_in, tmp_out)
end

function jprod_residual!(Jv::AbstractVector{T}, fd::ForwardDiffAD{T}, x::AbstractVector{T}, v::AbstractVector{T}, args...) where T
  SparseDiffTools.auto_jacvec!(Jv, fd.r!, x, v, fd.tmp_in, fd.tmp_out)
  Jv
end
=#

using SparseDiffTools, ReverseDiff, ForwardDiff

struct OptimizedForwardDiffADJprod{T} <: ADNLPModels.InPlaceADbackend
  r!
  tmp_in::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  tmp_out::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
end

function OptimizedForwardDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(n),
  kwargs...,
) where {T}
  tmp_in = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  tmp_out = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, ncon)
  return OptimizedForwardDiffADJprod(c!, tmp_in, tmp_out)
end

function ADNLPModels.Jprod(b::OptimizedForwardDiffADJprod, c!, x, v)
  ncon = length(b.tmp_out)
  Jv = similar(x, ncon)
  SparseDiffTools.auto_jacvec!(Jv, b.r!, x, v, b.tmp_in, b.tmp_out)
  @show ncon length(Jv) length(x)
  return Jv
end

struct OptimizedReverseDiffADJprod{T} <: ADNLPModels.InPlaceADbackend
  ϕ!
  tmp_in::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  tmp_out::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  z::Vector{T}
end

function OptimizedReverseDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(n),
  kwargs...,
) where {T}
  # ... auxiliary function for J(x) * v
  # ... J(x) * v is the derivative at t = 0 of t ↦ r(x + tv)
  ϕ!(out, t, x, v, tmp_in) = begin
    # here t is a vector of ReverseDiff.TrackedReal
    tmp_in .= x .+ t[1] .* v
    c!(out, tmp_in)
    out
  end
  tmp_in = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, nvar)
  tmp_out = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, ncon)
  return OptimizedReverseDiffADJprod(c!, tmp_in, tmp_out, [zero(T)])
end

function ADNLPModels.Jprod(b::OptimizedReverseDiffADJprod, c!, x, v)
  ncon = length(b.tmp_out)
  Jv = similar(x, ncon)
  ReverseDiff.jacobian!(Jv, (out, t) -> b.ϕ!(out, t, x, v, b._tmp_input), b.tmp_out, b.z)
  return Jv
end

benchmarked_optimized_backends["jprod_backend"] = Dict(
  "forward" => OptimizedForwardDiffADJprod,
  #"reverse" => OptimizedReverseDiffADJprod,
)

########################################################
# There are 6 levels:
# - bench-type (see `benchs`);
# - problem set (see `keys(problem_sets)`);
# - backend name (see `values(tested_backs)`);
# - backend (see `set_back_list(Val(f), test_back)`)
problem_sets = Dict(
  #"all" => setdiff(all_cons_problems, ["camshape"]), # crash
  "scalable" => scalable_cons_problems,
)
benchs = [
  "optimized",
  #"generic",
]
data_types = [Float64] # [Float16, Float32, Float64]
tested_backs = Dict(
  "jprod_backend" => :jprod,
)
const nscal = nn
name = "$(today())_adnlpmodels_benchmark_jprod"
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
              v = [sin(T(i) / 10) for i=1:n]
              SUITE[f][s][back_name][backend][T][pb] = @benchmarkable jprod(nlp, get_x0(nlp), $v) setup=(nlp = set_problem($pb, $(test_back), $backend, $(f), $s, $nscal, $T))
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
