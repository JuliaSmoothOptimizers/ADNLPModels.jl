using Pkg
Pkg.activate("benchmark_analyzer")
Pkg.instantiate()
using BenchmarkTools, Logging, JLD2, Dates
using StatsPlots

# BenchmarkTools.load("$name.json")
# plots
# using StatsPlots
# plot(t) # ou can use all the keyword arguments from Plots.jl, for instance st=:box or yaxis=:log10.
