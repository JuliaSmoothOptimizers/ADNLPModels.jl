#=
In this script, we benchmark several AD-backend.

TODO:
- automate "prepare benchmark" step for more functions
- analyze result
=#
using Pkg
Pkg.activate(".")
Pkg.add(url = "https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl", rev = "sparse-jc")
using ADNLPModels

using Dates, DelimitedFiles, JLD2, LinearAlgebra, Printf, SparseArrays
# using Pkg.Artifacts
using BenchmarkTools, DataFrames, JuMP, Plots
#JSO packages
using NLPModels, BenchmarkProfiles, NLPModelsJuMP, OptimizationProblems, SolverBenchmark
#This package
using ReverseDiff, Zygote, ForwardDiff

function get_optimized_list(optimized_backend)
  return union(keys(optimized_backend), [:jump])
end

is_jump_available(::Val{:jump}, T) = (T == Float64)
is_jump_available(::Val, T) = true

const meta = OptimizationProblems.meta
const nn = OptimizationProblems.default_nvar # 100 # default parameter for scalable problems

# Scalable problems from OptimizationProblem.jl
scalable_problems = meta[meta.variable_nvar .== true, :name][1:10] # problems that are scalable
all_problems = meta[meta.nvar .> 5, :name] # all problems with ≥ 5 variables
all_problems = setdiff(all_problems, scalable_problems)[1:10] # avoid duplicate problems

problem_sets = Dict(
  :all => all_problems,
  :scalable => scalable_problems,
)

# Available backends in ADNLPModels
# - ForwardDiffADGradient: optimized backend
# - ReverseDiffADGradient: optimized backend
# - ZygoteADGradient: generic

include("additional_backends.jl")

# Additional backends
# - GenericForwardDiffADGradient: generic
# - GenericReverseDiffADGradient: generic

###################################################
#
#
#
###################################################
benchmarked_optimized_backends = Dict(
  :gradient_backend => Dict(
    :forward => ADNLPModels.ForwardDiffADGradient,
    :reverse => ADNLPModels.ReverseDiffADGradient,
  ),
  :hprod_backend => Dict(),
  :jprod_backend => Dict(),
  :jtprod_backend => Dict(),
  :jacobian_backend => Dict(
    :sparse => ADNLPModels.SparseForwardADJacobian,
    :sym => ADNLPModels.SparseADJacobian,
  ),
  :hessian_backend => Dict(),
  :ghjvprod_backend => Dict(),
)

###################################################
#
#
#
###################################################
benchmarked_generic_backends = Dict(
  :gradient_backend => Dict(
    :forward => GenericForwardDiffADGradient,
    :reverse => GenericReverseDiffADGradient,
    :zygote => ADNLPModels.ZygoteADGradient,
  ),
  :hprod_backend => Dict(
    :forward => ADNLPModels.ForwardDiffADHvprod,
    :reverse => ADNLPModels.ReverseDiffADHvprod,
  ),
  :jprod_backend => Dict(
    :forward => ADNLPModels.ForwardDiffADJprod,
    :reverse => ADNLPModels.ReverseDiffADJprod,
    :zygote => ADNLPModels.ZygoteADJprod,
  ),
  :jtprod_backend => Dict(
    :forward => ADNLPModels.ForwardDiffADJtprod,
    :reverse => ADNLPModels.ReverseDiffADJtprod,
    :zygote => ADNLPModels.ZygoteADJtprod,
  ),
  :jacobian_backend => Dict(
    :forward => ADNLPModels.ForwardDiffADJacobian,
    :reverse => ADNLPModels.ReverseDiffADJacobian,
    :zygote => ADNLPModels.ZygoteADJacobian,
  ),
  :hessian_backend => Dict(
    :forward => ADNLPModels.ForwardDiffADHessian,
    :reverse => ADNLPModels.ReverseDiffADHessian,
    :zygote => ADNLPModels.ZygoteADHessian,
  ),
  :ghjvprod_backend => Dict(
    :forward => ADNLPModels.ForwardDiffADGHjvprod,
  ),
)

function set_back_list(::Val{:optimized}, test_back::Symbol)
  return get_optimized_list(benchmarked_optimized_backends[test_back])
end

function get_back(::Val{:optimized}, test_back::Symbol, backend::Symbol)
  # test_back must be a key in benchmarked_optimized_backends
  # backend must be a key in benchmarked_optimized_backends[test_back]
  return benchmarked_optimized_backends[test_back][backend]
end

function set_back_list(::Val{:generic}, test_back::Symbol)
  return keys(benchmarked_generic_backends[test_back])
end

function get_back(::Val{:generic}, test_back::Symbol, backend::Symbol)
  # test_back must be a key in benchmarked_generic_backends
  # backend must be a key in benchmarked_generic_backends[test_back]
  return benchmarked_generic_backends[test_back][backend]
end

# keys list all the accepted keywords to define backends
# values are generic backend to be used by default in this benchmark
all_backend_structure = Dict(
  :gradient_backend => GenericForwardDiffADGradient,
  :hprod_backend => ADNLPModels.ForwardDiffADHvprod,
  :jprod_backend => ADNLPModels.ForwardDiffADJprod,
  :jtprod_backend => ADNLPModels.ForwardDiffADJtprod,
  :jacobian_backend => ADNLPModels.ForwardDiffADJacobian,
  :hessian_backend => ADNLPModels.ForwardDiffADHessian,
  :ghjvprod_backend => ADNLPModels.ForwardDiffADGHjvprod,
)

"""
Return an ADNLPModel with `back_struct` as an AD backend for `test_back ∈ keys(all_backend_structure)`
"""
function set_adnlp(pb::String, test_back::Symbol, back_struct::Type{<:ADNLPModels.ADBackend}, n::Integer = nn, T::DataType = Float64)
  pbs = Meta.parse(pb)
  backend_structure = Dict{Symbol, Any}()
  for k in keys(all_backend_structure)
    if k == test_back
      push!(backend_structure, k => back_struct)
    else
      push!(backend_structure, k => all_backend_structure[k])
    end
  end
  return OptimizationProblems.ADNLPProblems.eval(pbs)(
    ;type = Val(T),
    n = n,
    gradient_backend = backend_structure[:gradient_backend],
    hprod_backend = backend_structure[:hprod_backend],
    jprod_backend = backend_structure[:jprod_backend],
    jtprod_backend = backend_structure[:jtprod_backend],
    jacobian_backend = backend_structure[:jacobian_backend],
    hessian_backend = backend_structure[:hessian_backend],
    ghjvprod_backend = backend_structure[:ghjvprod_backend],
  )
end

function set_adnlp(pb::String, f::Symbol, test_back::Symbol, backend::Symbol, n::Integer = nn, T::DataType = Float64)
  back_struct = get_back(Val(f), test_back, backend)
  return set_adnlp(pb, test_back, back_struct, n, T)
end

function set_problem(pb::String, test_back::Symbol, backend::Symbol, f::Symbol, s::Symbol, n::Integer = nn, T::DataType = Float64)
  nlp = if backend == :jump
    model = if s == :scalable
      OptimizationProblems.PureJuMP.eval(Meta.parse(pb))(n)
    else
      OptimizationProblems.PureJuMP.eval(Meta.parse(pb))()
    end
    MathOptNLPModel(model)
  else
    set_adnlp(pb, f, test_back, backend, n, T)
  end
  return nlp
end

########################################################
# There are 6 levels:
# - bench-type (see `benchs`);
# - problem set (see `keys(problem_sets)`);
# - backend name (see `values(tested_backs)`);
# - backend (see `set_back_list(Val(f), test_back)`)
benchs = [:optimized, :generic]
data_types = [Float16, Float32, Float64]
tested_backs = Dict(
  :gradient_backend => :grad!,
  :jacobian_backend => :jac_coord!,
)
########################################################

@info "Initialize benchmark"
const SUITE = BenchmarkGroup()

for f in benchs
  SUITE[f] = BenchmarkGroup()
  for s in keys(problem_sets)
    SUITE[f][s] = BenchmarkGroup()
    for test_back in keys(tested_backs)
      back_name = tested_backs[test_back]
      SUITE[f][s][back_name] = BenchmarkGroup()
      for backend in set_back_list(Val(f), test_back)
        SUITE[f][s][back_name][backend] = BenchmarkGroup()
        for T in data_types
          if is_jump_available(Val(backend), T)
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

const nscal = nn * 10

@info "Prepare benchmark"
for f in benchs
  for s in keys(problem_sets)

    test_back = :gradient_backend
    back_name = tested_backs[test_back]
    back_list = set_back_list(Val(f), test_back)
    for backend in back_list
      problems = problem_sets[s]
      for T in data_types
        if !(backend == :jump && T != Float64)
          for pb in problems
            nlp = set_problem(pb, test_back, backend, f, s, nscal, T)
            # add some asserts to make sure it is ok
            # @info " $(pb): $(eltype(nlp.meta.x0)) with $(nlp.meta.nvar) vars and $(nlp.meta.ncon) cons"
            x = nlp.meta.x0
            g = similar(x)
            SUITE[f][s][back_name][backend][T][pb] = @benchmarkable grad!($nlp, $x, $g)
          end
        end
      end
    end
#=
    test_back = :jacobian_backend
    back_name = tested_backs[test_back]
    back_list = set_back_list(Val(f), test_back)
    for backend in back_list
      problems = problem_sets[s]
      for T in data_types
        if !(backend == :jump && T != Float64)
          for pb in problems
            nlp = set_problem(pb, test_back, backend, f, s, nscal, T)
            # add some asserts to make sure it is ok
            # @info " $(pb): $(eltype(nlp.meta.x0)) with $(nlp.meta.nvar) vars and $(nlp.meta.ncon) cons"
            x = nlp.meta.x0
            vals = similar(x, nlp.meta.nnzj)
            SUITE[f][s][back_name][backend][T][pb] = @benchmarkable jac_coord!($nlp, $x, $vals)
          end
        end
      end
    end
=#
  end
end

@info "Starting evaluating the benchmark"
result = run(SUITE)

@save "$(today())_nscal_$(nscal)_adnlpmodels_benchmark.jld2" result
