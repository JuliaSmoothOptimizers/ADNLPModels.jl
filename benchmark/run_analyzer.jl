using Pkg
Pkg.activate("benchmark/benchmark_analyzer")
Pkg.instantiate()
using BenchmarkTools, Dates, JLD2, JSON, Plots, StatsPlots

# BenchmarkTools.load("$name.json")
# plots
# using StatsPlots
# plot(t) # ou can use all the keyword arguments from Plots.jl, for instance st=:box or yaxis=:log10.
