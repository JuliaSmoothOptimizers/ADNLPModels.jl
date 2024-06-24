using Pkg, Logging, JLD2, Dates
Pkg.activate("benchmark")
# instantiate
# up ADNLPModels

include("benchmarks.jl")

@info "TUNE"
@time with_logger(ConsoleLogger(Error)) do # remove warnings
  tune!(SUITE)
end

@info "RUN"
@time result = with_logger(ConsoleLogger(Error)) do # remove warnings
  run(SUITE)
end

@info "SAVE BENCHMARK RESULT"
name = "$(today())_adnlpmodels_benchmark"
@save "$name.jld2" result

@info "ANALYZE"
# save the result in a jld2 file?
# plots
