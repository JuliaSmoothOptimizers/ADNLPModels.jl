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

include("utils.jl")

########################################################
# There are 6 levels:
# - bench-type (see `benchs`);
# - problem set (see `keys(problem_sets)`);
# - backend name (see `values(tested_backs)`);
# - backend (see `set_back_list(Val(f), test_back)`)
problem_sets = Dict(
  :all => all_cons_problems,
  #:scalable => scalable_cons_problems,
)
benchs = [
  :optimized,
  #:generic,
]
data_types = [Float64] # [Float16, Float32, Float64]
tested_backs = Dict(
  :jacobian_backend => :jac_coord!,
)
const nscal = nn * 10
name = "$(today())_adnlpmodels_benchmark_jac"
if :all in keys(problem_sets)
  name *= "_all"
end
if :scalable in keys(problem_sets)
  name *= "_nscal_$(nscal)"
end
if data_types == [Float64]
  name *= "_mono"
else
  name *= "_multi"
end
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

    test_back = :jacobian_backend
    back_name = tested_backs[test_back]
    back_list = set_back_list(Val(f), test_back)
    for backend in back_list
      problems = problem_sets[s]
      for T in data_types
        @info "Run $backend with T=$T"
        if !(backend == :jump && T != Float64)
          for pb in problems
            nlp = set_problem(pb, test_back, backend, f, s, nscal, T)
            # add some asserts to make sure it is ok
            @info " $(pb): $(eltype(nlp.meta.x0)) with $(nlp.meta.nvar) vars and $(nlp.meta.ncon) cons"
            x = nlp.meta.x0
            vals = similar(x, nlp.meta.nnzj)
            SUITE[f][s][back_name][backend][T][pb] = @benchmarkable jac_coord!($nlp, $x, $vals)
          end
        end
      end
    end
  end
end

@info "Starting evaluating the benchmark"
result = run(SUITE)

@save "$name.jld2" result
