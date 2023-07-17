#=
In this script, we benchmark several AD-backend.

TODO:
- automate "prepare benchmark" step for more functions
- analyze result
=#
using Pkg
Pkg.activate(".")
Pkg.add(url = "https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl", rev = "reverseforwardhpro")
using ADNLPModels

using Dates, DelimitedFiles, JLD2, LinearAlgebra, Printf, SparseArrays
# using Pkg.Artifacts
using BenchmarkTools, DataFrames, JuMP, Plots
#JSO packages
using NLPModels, BenchmarkProfiles, NLPModelsJuMP, OptimizationProblems, SolverBenchmark
#This package
using SparseDiffTools, ReverseDiff, Zygote, ForwardDiff

include("additional_backends.jl")
include("utils.jl")

using ReverseDiff, ForwardDiff

#=
julia> scalable_cons_problems
12-element Vector{String}:
 "camshape": Cannot determine ordering of Dual tags ForwardDiff.Tag{ReverseDiff.var"#129#132"{typeof(*), ForwardDiff.Dual{ForwardDiff.Tag{OptimizationProblems.ADNLPProblems.var"#f#273"{Int64, Int64}, Float64}, Float64, 1}}, ForwardDiff.Dual{ForwardDiff.Tag{OptimizationProblems.ADNLPProblems.var"#f#273"{Int64, Int64}, Float64}, Float64, 1}} and ForwardDiff.Tag{OptimizationProblems.ADNLPProblems.var"#f#273"{Int64, Int64}, Float64}
 "chain": OK
 "channel": OK
 "clnlbeam": OK
 "controlinvestment": OK
 "elec": OK
 "hovercraft1d": OK 
 "polygon": OK 
 "polygon1": OK 
 "polygon3": OK 
 "robotarm": OK 
 "structural": Cannot determine ordering of Dual tags ForwardDiff.Tag{ReverseDiff.var"#129#132"{typeof(*), ForwardDiff.Dual{ForwardDiff.Tag{OptimizationProblems.ADNLPProblems.var"#f#1844"{OptimizationProblems.ADNLPProblems.var"#f#1838#1845"{Vector{Float64}}, Int64}, Float64}, Float64, 1}}, ForwardDiff.Dual{ForwardDiff.Tag{OptimizationProblems.ADNLPProblems.var"#f#1844"{OptimizationProblems.ADNLPProblems.var"#f#1838#1845"{Vector{Float64}}, Int64}, Float64}, Float64, 1}} and ForwardDiff.Tag{OptimizationProblems.ADNLPProblems.var"#f#1844"{OptimizationProblems.ADNLPProblems.var"#f#1838#1845"{Vector{Float64}}, Int64}, Float64}

=#

########################################################
# There are 6 levels:
# - bench-type (see `benchs`);
# - problem set (see `keys(problem_sets)`);
# - backend name (see `values(tested_backs)`);
# - backend (see `set_back_list(Val(f), test_back)`)
problem_sets = Dict(
  #"all" => setdiff(all_cons_problems, ["camshape"]), # crash
  "scalable" => setdiff(scalable_cons_problems, ["camshape", "clnlbeam", "polygon1", "structural"]), # , ["polygon", "polygon1", "clnlbeam"]
)
benchs = [
  "optimized",
  #"generic",
]
data_types = [Float64] # [Float16, Float32, Float64]
tested_backs = Dict(
  "hprod_backend" => :hprod,
)
const nscal = nn * 1
name = "$(today())_adnlpmodels_benchmark_hprod_lag"
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
              SUITE[f][s][back_name][backend][T][pb] = @benchmarkable hprod(nlp, get_x0(nlp), get_y0(nlp), $v) setup=(nlp = set_problem($pb, $(test_back), $backend, $(f), $s, $nscal, $T))
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
