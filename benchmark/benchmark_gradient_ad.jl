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
using  ReverseDiff, Zygote, ForwardDiff # Enzyme,

include("additional_backends.jl")
include("utils.jl")

########################################################
# There are 6 levels:
# - bench-type (see `benchs`);
# - problem set (see `keys(problem_sets)`);
# - backend name (see `values(tested_backs)`);
# - backend (see `set_back_list(Val(f), test_back)`)
problems_not_supported_enzyme = [
  "brybnd",
  "clplatea",
  "clplateb",
  "clplatec",
  "curly",
  "curly10",
  "curly20",
  "curly30",
  "elec",
  "fminsrf2",
  "hs101",
  "hs117",
  "hs119",
  "hs86",
  "integreq",
  "ncb20",
  "ncb20b",
  "palmer1c",
  "palmer1d",
  "palmer2c",
  "palmer3c",
  "palmer4c",
  "palmer5c",
  "palmer5d",
  "palmer6c",
  "palmer7c",
  "palmer8c",
  "sbrybnd",
  "tetra",
  "tetra_duct12",
  "tetra_duct15",
  "tetra_duct20",
  "tetra_foam5",
  "tetra_gear",
  "tetra_hook",
  "threepk",
  "triangle",
  "triangle_deer",
  "triangle_pacman",
  "triangle_turtle",
  "watson",
]
problem_sets = Dict(
  #"all" => all_problems,
  "scalable" => scalable_problems, # setdiff(scalable_problems, problems_not_supported_enzyme),
)
benchs = [
  "optimized",
  #"generic",
]
data_types = [Float64] # [Float16, Float32, Float64]
tested_backs = Dict(
  "gradient_backend" => :grad!,
)
const nscal = nn * 10
name = "$(today())_adnlpmodels_benchmark_grad"
if "all" in keys(problem_sets)
  name *= "_all"
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

    test_back = "gradient_backend"
    back_name = tested_backs[test_back]
    back_list = set_back_list(Val(Symbol(f)), test_back)
    for backend in back_list
      problems = problem_sets[s]
      for T in data_types
        if !(backend == "jump" && T != Float64)
          for pb in problems
            #nlp = set_problem(pb, test_back, backend, f, s, nscal, T)
            # add some asserts to make sure it is ok
            n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
            m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
            @info " $(pb): $T with $n vars and $m cons"
            # x = nlp.meta.x0
            g = zeros(T, n)
            SUITE[f][s][back_name][backend][T][pb] = @benchmarkable grad!(nlp, get_x0(nlp), $g) setup=(nlp = set_problem($pb, $(test_back), $backend, $(f), $s, $nscal, $T))
          end
        end
      end
    end
  end
end

#=
# If a cache of tuned parameters already exists, use it, otherwise, tune and cache
# the benchmark parameters. Reusing cached parameters is faster and more reliable
# than re-tuning `suite` every time the file is included.
paramspath = joinpath(dirname(@__FILE__), "params.json")

if isfile(paramspath)
    loadparams!(suite, BenchmarkTools.load(paramspath)[1], :evals);
else
    tune!(suite)
    BenchmarkTools.save(paramspath, params(suite));
end
=#

@info "Starting evaluating the benchmark"
result = run(SUITE)

@save "$name.jld2" result
