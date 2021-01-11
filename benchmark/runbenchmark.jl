using BenchmarkTools, DataFrames, Plots, Profile
#using ProfileView
#JSO packages
using NLPModels, SolverBenchmark
#This package
using ADNLPModels

#gr()

problems = ["hs5", "brownden"]
problems2 = ["arglina", "arglinb", "arglinc", "arwhead", "bdqrtic", "beale", "broydn7d",
             "brybnd", "chainwoo", "chnrosnb", "cosine", "cragglvy", "dixon3dq", "dqdrtic",
             "dqrtic", "edensch", "eg2", "engval1", "errinros", "extrosnb", "fletcbv2",
             "fletcbv3", "fletchcr", "freuroth", "genhumps", "genrose", "genrose_nash",
             "indef", "liarwhd", "morebv", "ncb20", "ncb20b", "noncvxu2", "noncvxun",
             "nondia", "nondquar", "nzf1", "penalty2", "penalty3", "powellsg", "power",
             "quartc", "sbrybnd", "schmvett", "scosine", "sparsine", "sparsqur", "srosenbr",
             "sinquad", "tointgss", "tquartic", "tridia", "vardim", "woods"]

#List of problems used in tests
#Problems from NLPModels
include("../test/problems/hs5.jl") #bounds constraints n=2, dense hessian
include("../test/problems/brownden.jl") #unconstrained n=4, dense hessian

for pb in union(problems, problems2)
    include("../test/problems/$(lowercase(pb)).jl")
end

test_problems_1 = [eval(Meta.parse("$(pb)_radnlp()")) for pb in problems]
test_problems_2 = [eval(Meta.parse("$(pb)_autodiff()")) for pb in problems]

###############################################################################
# TO COPY-PASTE and modify.
func = nlp -> obj(nlp, nlp.meta.x0)

name1 = "RADNLPModel"
nlp = test_problems_1[1]
a1 = @benchmark func(nlp)
name2 = "ADNLPModel"
nlp = test_problems_2[1]
a2 = @benchmark func(nlp)
###############################################################################

###############################################################################
# TO COPY-PASTE and modify.
# Run a benchmark of the functions on each problem
# TODO: how to treat the result?
# 
#
function runbenchmark(problems)
  #with BenchmarkTools
  #Example: https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/benchmark/benchmarks.jl
  suite = BenchmarkGroup()

  # Add some child groups to our benchmark suite.
  suite["obj"] = BenchmarkGroup()
  suite["grad"] = BenchmarkGroup()
  
  x_tab = ["x0", "xrand"]

  for pb in problems
    npb1 = eval(Meta.parse("$(pb)_radnlp()"))
    npb2 = eval(Meta.parse("$(pb)_autodiff()"))
    k=1
    for x in (npb1.meta.x0, rand(npb1.meta.nvar))
        suite["obj"][string(pb), x_tab[k], "radnlp"] = @benchmarkable obj($npb1, $x)
        suite["obj"][string(pb), x_tab[k], "radnlp"] = @benchmarkable obj($npb2, $x)
        suite["grad"][string(pb), x_tab[k], "radnlp"] = @benchmarkable grad($npb1, $x)
        suite["grad"][string(pb), x_tab[k], "radnlp"] = @benchmarkable grad($npb2, $x)
        k+=1
    end
  end

  return run(suite)
end
###############################################################################