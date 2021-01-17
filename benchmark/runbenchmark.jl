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
#include("../test/problems/hs5.jl") #bounds constraints n=2, dense hessian
#include("../test/problems/brownden.jl") #unconstrained n=4, dense hessian

for pb in union(problems, problems2)
    include("../test/problems/$(lowercase(pb)).jl")
end

include("additional_func.jl")

models = [:radnlp, :autodiff]
fun    = [:obj, :grad]

rb = runbenchmark(problems, models, fun)
N = length(rb[fun[1]][models[1]]) #number of problems
gstats = group_stats(rb, N, fun, models)

for f in fun
  cost(df) = df.mean_time
  p = performance_profile(gstats[f], cost)
  png("perf-$(f)")
end