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
using ReverseDiff, SparseDiffTools, Zygote, ForwardDiff

include("additional_backends.jl")
include("utils.jl")

########################################################
# There are 6 levels:
# - bench-type (see `benchs`);
# - problem set (see `keys(problem_sets)`);
# - backend name (see `values(tested_backs)`);
# - backend (see `set_back_list(Val(f), test_back)`)
problem_sets = Dict(
  #"all" => all_cons_problems, # crash
  "scalable" => setdiff(scalable_cons_problems, ["polygon", "camshape"]),
)
benchs = [
  "optimized",
  #"generic",
]
data_types = [Float64] # [Float16, Float32, Float64]
tested_backs = Dict(
  "hessian_backend" => :hess_coord!,
)
const nscal = nn * 10
name = "$(today())_adnlpmodels_benchmark_hess"
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

    test_back = "hessian_backend"
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
            # x = nlp.meta.x0
            #vals = similar(x, nlp.meta.nnzj)
            #SUITE[f][s][back_name][backend][T][pb] = @benchmarkable grad!(nlp, get_x0(nlp), $g) setup=(nlp = set_problem($pb, $(test_back), $backend, $(f), $s, $nscal, $T))
            #SUITE[f][s][back_name][backend][T][pb] = @benchmarkable hess_coord!($nlp, $x, $vals)
            SUITE[f][s][back_name][backend][T][pb] = @benchmarkable hess_coord(nlp, get_x0(nlp), get_y0(nlp)) setup=(nlp = set_problem($pb, $(test_back), $backend, $(f), $s, $nscal, $T))
          end
        end
      end
    end
  end
end

@info "Starting evaluating the benchmark"
result = run(SUITE)

@save "$name.jld2" result

#=

for problem in scalable_cons_problems
  nlp = OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))()
  n = nlp.meta.nvar
  b = SparseForwardADHessian(n, nlp.f, nlp.meta.nnln, nlp.c!)
  nnzh = length(b.rowval)
  nlp_ju = MathOptNLPModel(OptimizationProblems.PureJuMP.eval(Meta.parse(problem))())
  @info "$problem nvar=$(n) ADnnzh=$(nnzh) ADpercentage=$(nnzh/(n * (n + 1))) JuMPnnzh=$(nlp_ju.meta.nnzh) JuMPpercentage=$(nlp_ju.meta.nnzh/(n * (n + 1)))"
  x = get_x0(nlp) .+ rand()
  y = get_y0(nlp) .+ rand()
  @show norm(hess(nlp, x, y) - hess(nlp_ju, x, y))
end

[ Info: camshape nnzh=595 nvar=100 percentage=0.05891089108910891
[ Info: chain nnzh=50 nvar=100 percentage=0.0049504950495049506
[ Info: chain nnzh=192 nvar=100 percentage=0.01900990099009901
[ Info: channel nnzh=384 nvar=96 percentage=0.041237113402061855
[ Info: channel nnzh=1728 nvar=96 percentage=0.18556701030927836
[ Info: clnlbeam nnzh=66 nvar=99 percentage=0.006666666666666667
[ Info: clnlbeam nnzh=130 nvar=99 percentage=0.013131313131313131
[ Info: controlinvestment nnzh=50 nvar=100 percentage=0.0049504950495049506
[ Info: controlinvestment nnzh=444 nvar=100 percentage=0.04396039603960396
[ Info: elec nnzh=4950 nvar=99 percentage=0.5
[ Info: elec nnzh=5049 nvar=99 percentage=0.51
[ Info: hovercraft1d nnzh=32 nvar=98 percentage=0.0032982890125747267
[ Info: hovercraft1d nnzh=32 nvar=98 percentage=0.0032982890125747267
[ Info: polygon nnzh=5050 nvar=100 percentage=0.5
[ Info: polygon nnzh=12596 nvar=100 percentage=1.2471287128712871
[ Info: polygon1 nnzh=300 nvar=100 percentage=0.0297029702970297
[ Info: polygon1 nnzh=350 nvar=100 percentage=0.034653465346534656
[ Info: polygon3 nnzh=200 nvar=100 percentage=0.019801980198019802
[ Info: polygon3 nnzh=600 nvar=100 percentage=0.0594059405940594
[ Info: robotarm nnzh=138 nvar=109 percentage=0.011509591326105087
[ Info: robotarm nnzh=306 nvar=109 percentage=0.025521267723102585
[ Info: structural nnzh=0 nvar=600 percentage=0.0
[ Info: structural nnzh=0 nvar=600 percentage=0.0
=#