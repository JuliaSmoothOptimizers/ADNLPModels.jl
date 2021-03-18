using BenchmarkTools, DataFrames, Plots, Profile
#using ProfileView
#JSO packages
using NLPModels, BenchmarkProfiles
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
#problems2 = ["arglina", "arglinb", "arglinc", "arwhead", "bdqrtic", "beale", "broydn7d"]
problems = union(problems, problems2)
#List of problems used in tests
#Problems from NLPModels
#include("../test/problems/hs5.jl") #bounds constraints n=2, dense hessian
#include("../test/problems/brownden.jl") #unconstrained n=4, dense hessian

for pb in problems
    include("../test/problems/$(lowercase(pb)).jl")
end

include("additional_func.jl")

#Extend the functions of each problems to the variants of RADNLPModel
for pb in problems #readdir("test/problems")
  eval(Meta.parse("$(pb)_radnlp_reverse(args... ; kwargs...) = $(pb)_radnlp(args... ; gradient = ADNLPModels.reverse, kwargs...)"))
  eval(Meta.parse("$(pb)_radnlp_smartreverse(args... ; kwargs...) = $(pb)_radnlp(args... ; gradient = ADNLPModels.smart_reverse, kwargs...)"))
end

models = [:radnlp_smartreverse, :autodiff]#[:radnlp_reverse, :radnlp_smartreverse, :autodiff]
fun    = [:obj, :grad]

rb = runbenchmark(problems, models, fun)
N = length(rb[fun[1]][models[1]]) #number of problems
gstats = group_stats(rb, N, fun, models)

function performance_profile(stats::Dict{Symbol,DataFrame}, cost::Function, args...; kwargs...)
  solvers = keys(stats)
  dfs = (stats[s] for s in solvers)
  P = hcat([cost(df) for df in dfs]...)
  performance_profile(P, string.(solvers), args...; kwargs...)
end

for f in fun
  cost(df) = df.mean_time
  p = performance_profile(gstats[f], cost)
  png("perf-$(f)")
end