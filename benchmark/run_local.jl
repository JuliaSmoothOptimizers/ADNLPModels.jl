using Pkg, Logging
Pkg.activate("benchmark")
# instantiate
# up ADNLPModels

include("benchmarks.jl")

@info "TUNE"
with_logger(ConsoleLogger(Error)) do
  tune!(SUITE)
end

@info "RUN"
run(SUITE)

@info "ANALYZE"
# save the result in a jld2 file?
# plots
