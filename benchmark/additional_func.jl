###############################################################################
# Run a benchmark of the functions on each problem
#
function runbenchmark(problems,
                      models,
                      fun)
  #with BenchmarkTools
  #Example: https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/benchmark/benchmarks.jl
  suite = BenchmarkGroup()

  # Add some child groups to our benchmark suite.
  for f in fun
    suite[f] = BenchmarkGroup()
    for m in models
      suite[f][m] = BenchmarkGroup()
    end
  end

  x_tab = ["x0", "xrand"]

  for pb in problems
    for m in models
      npb = eval(Meta.parse("$(pb)_$(m)()"))
      for f in fun
        k = 1
        for x in (npb.meta.x0, rand(npb.meta.nvar))
          suite[f][m][string(pb), x_tab[k]] = @benchmarkable eval($f)($npb, $x)
          k += 1
        end
      end
    end
  end

  return run(suite)
end

###############################################################################

function groups_to_stats(rb :: BenchmarkTools.BenchmarkGroup, N, models)
  empty_df = DataFrame(pb        = [k for k in keys(rb[models[1]].data)],
                       memory    = [NaN for i=1:N],
                       allocs    = [NaN for i=1:N],
                       max_time  = [NaN for i=1:N],
                       mean_time = [NaN for i=1:N],
                       )
  stats = Dict([m => copy(empty_df) for m in models])
  for m in models
    collec = [s for (r,s) in rb[m]]
    stats[m][!,"memory"] = [c.memory for c in collec]
    stats[m][!,"allocs"] = [c.allocs for c in collec]
    stats[m][!,"max_time"] = [maximum(c.times) for c in collec]
    stats[m][!,"mean_time"] = [sum(c.times)/c.params.samples for c in collec]
  end
  return stats
end

function group_stats(rb :: BenchmarkTools.BenchmarkGroup, N, fun, models)
  gstats = Dict([Symbol(f) => groups_to_stats(rb[f], N, models) for f in fun])
end

function performance_profile(stats::Dict{Symbol,DataFrame}, cost::Function, args...; kwargs...)
  solvers = keys(stats)
  dfs = (stats[s] for s in solvers)
  P = hcat([cost(df) for df in dfs]...)
  SolverBenchmark.performance_profile(P, string.(solvers), args...; kwargs...)
end
