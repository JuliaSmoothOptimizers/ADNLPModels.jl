#=
INTRODUCTION OF THIS BENCHMARK:

We test here the function `hess_residual_coord!` for ADNLPModels with different backends:
  - ADNLPModels.SparseADJacobian
  - ADNLPModels.SparseReverseADHessian
=#
using ForwardDiff, SparseConnectivityTracer, SparseMatrixColorings

include("additional_backends.jl")

data_types = [Float32, Float64]

benchmark_list = [:optimized]

benchmarked_hessian_backend = Dict(
  "sparse" => ADNLPModels.SparseADHessian,
  #"sparse-reverse" => ADNLPModels.SparseReverseADHessian, #failed
)
get_backend_list(::Val{:optimized}) = keys(benchmarked_hessian_backend)
get_backend(::Val{:optimized}, b::String) = benchmarked_hessian_backend[b]

problem_sets = Dict("scalable_nls" => scalable_nls_problems)
nscal = 1000

name_backend = "hessian_residual_backend"
fun = hess_coord_residual
@info "Initialize $(fun) benchmark"
SUITE["$(fun)"] = BenchmarkGroup()

for f in benchmark_list
  SUITE["$(fun)"][f] = BenchmarkGroup()
  for T in data_types
    SUITE["$(fun)"][f][T] = BenchmarkGroup()
    for s in keys(problem_sets)
      SUITE["$(fun)"][f][T][s] = BenchmarkGroup()
      for b in get_backend_list(Val(f))
        SUITE["$(fun)"][f][T][s][b] = BenchmarkGroup()
        backend = get_backend(Val(f), b)
        for pb in problem_sets[s]
          n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
          m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
          nequ = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nls_nequ(n = $(nscal))"))
          verbose_subbenchmark && @info " $(pb): $T with $n vars, $nequ residuals and $m cons"
          v = 10 * T[-(-1.0)^i for i = 1:nequ]
          SUITE["$(fun)"][f][T][s][b][pb] = @benchmarkable $fun(nls, get_x0(nls), $v) setup =
            (nls = set_adnls($pb, $(name_backend), $backend, $nscal, $T))
        end
      end
    end
  end
end
