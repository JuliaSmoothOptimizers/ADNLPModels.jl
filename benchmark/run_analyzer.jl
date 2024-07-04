using Pkg
Pkg.activate("benchmark/benchmark_analyzer")
Pkg.instantiate()
using BenchmarkTools, Dates, JLD2, JSON, Plots, StatsPlots

# name of the result file:
name = ""
resultpath = joinpath(dirname(@__FILE__), "results")
if name == ""
  name = replace(readdir(resultpath)[end], ".jld2" => "", ".json" => "")
end

@load joinpath(dirname(@__FILE__), "results", "$name.jld2") result
t = BenchmarkTools.load(joinpath(dirname(@__FILE__), "results", "$name.json"))

# plots
using StatsPlots
plot(t) # ou can use all the keyword arguments from Plots.jl, for instance st=:box or yaxis=:log10.

@info "Available benchmarks"
df_results = Dict{String, Dict{Symbol, DataFrame}}()
for benchmark in keys(result)
  result_bench = result[benchmark] # one NLPModel API function
  for benchmark_list in keys(result_bench)
    for type_bench in keys(result_bench[benchmark_list])
      for set_bench in keys(result_bench[benchmark_list][type_bench])
        @info "$benchmark/$benchmark_list for type $type_bench on problem set $(set_bench)"
        bench = result_bench[benchmark_list][type_bench][set_bench]
        df_results["$(benchmark)_$(benchmark_list)_$(type_bench)_$(set_bench)"] = bg_to_df(bench)
      end
    end
  end
end

function bg_to_df(bench::BenchmarkGroup)
  solvers = collect(keys(bench)) # "jump", ...
  nsolvers = length(solvers)
  problems = collect(keys(bench[solvers[1]]))
  nprob = length(problems)
  dfT = Dict{Symbol, DataFrame}()
  for solver in solvers
    dfT[Symbol(solver)] = DataFrame(
      [
        [median(bench[solver][pb]).time for pb in problems],
        [median(bench[solver][pb]).memory for pb in problems],
      ],
      [:median_time, :median_memory],
    )
  end
  return dfT
end

using SolverBenchmark, BenchmarkProfiles

# b::BenchmarkProfiles.AbstractBackend = PlotsBackend()
costs = [df -> df.median_time, df -> df.median_memory]
costnames = ["median time", "median memory"]
for key_benchmark in keys(df_results)
  stats = df_results[key_benchmark]
  p = profile_solvers(stats, costs, costnames)
  savefig(p, "$(name)_$(key_benchmark).png")
end
