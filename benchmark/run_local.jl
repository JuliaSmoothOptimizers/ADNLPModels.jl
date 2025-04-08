using Pkg
Pkg.activate("benchmark")
Pkg.instantiate()
Pkg.update("ADNLPModels")
using Logging, JLD2, Dates

path = dirname(@__FILE__)
skip_tune = true

@info "INITIALIZE"
include("benchmarks.jl")

list_of_benchmark = keys(SUITE)
# gradient: SUITE[@tagged "grad!"]
# Coloring benchmark: SUITE[@tagged "hessian_backend" || "hessian_residual_backend" || "jacobian_backend" || "jacobian_residual_backend"]
# Matrix benchmark: SUITE[@tagged "hessian_backend" || "hessian_residual_backend" || "jacobian_backend" || "jacobian_residual_backend" || "hess_coord!" || "hess_coord_residual!" || "jac_coord!" || "jac_coord_residual!"]
# Matrix-vector products: SUITE[@tagged "hprod!" || "hprod_residual!" || "jprod!" || "jprod_residual!" || "jtprod!" || "jtprod_residual!"]

for benchmark_in_suite in list_of_benchmark
  @info "$(benchmark_in_suite)"
end

@info "TUNE"
if !skip_tune
  @time with_logger(ConsoleLogger(Error)) do
    tune!(SUITE)
    BenchmarkTools.save("params.json", params(suite))
  end
else
  @info "Skip tuning"
  # https://juliaci.github.io/BenchmarkTools.jl/dev/manual/
  BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
end

@info "RUN"
@time result = with_logger(ConsoleLogger(Error)) do     # remove warnings
  if "params.json" in (path == "" ? readdir() : readdir(path))
    loadparams!(suite, BenchmarkTools.load("params.json")[1], :evals, :samples)
  end
  run(SUITE, verbose = true)
end

@info "SAVE BENCHMARK RESULT"
name = "$(today())_adnlpmodels_benchmark"
@save "$name.jld2" result
BenchmarkTools.save("$name.json", result)
