using Pkg; Pkg.activate(".")
using JLD2, BenchmarkTools, DataFrames, Plots, Dates

name = "2023-05-10_adnlpmodels_benchmark_jprod_optimized_nscal_1000_mono"
@load "$name.jld2" result

# track the main table:
global temp = result
global k = keys(result)
while (length(k) == 1)
  global temp = temp[first(k)]
  global k = keys(temp)
end

solvers = collect(keys(temp)) # "jump", ...
nsolvers = length(solvers)
types = collect(keys(temp[solvers[1]]))
ntypes = length(types)
problems = collect(keys(temp[solvers[1]][types[1]]))
nprob = length(problems)

DF = Dict{Symbol, Any}()
for T in types
  dfT = Dict{Symbol, DataFrame}()
  for solver in solvers
    dfp = DataFrame(
      [
        [median(temp[solver][T][pb]).time for pb in problems],
        [median(temp[solver][T][pb]).memory for pb in problems],
      ],
      [:median_time, :median_memory]
    )
    dfT[Symbol(solver)] = dfp
  end
  DF[Symbol(T)] = dfT
end

using SolverBenchmark, BenchmarkProfiles

# b::BenchmarkProfiles.AbstractBackend = PlotsBackend()
stats = DF[:Float64]
costs =[
  df -> df.median_time,
  df -> df.median_memory,
]
costnames = ["median time", "median memory"]
p = profile_solvers(stats, costs, costnames)
savefig(p, name * ".png")
