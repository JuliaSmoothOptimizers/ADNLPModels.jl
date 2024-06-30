#=
INTRODUCTION OF THIS BENCHMARK:

We test here the function `jac_coord!` for ADNLPModels with different backends:
  - ADNLPModels.SparseADJacobian
=#
using ForwardDiff, SparseConnectivityTracer, SparseMatrixColorings

include("additional_backends.jl")

data_types = [Float32, Float64]

benchmark_list = [:optimized]

benchmarked_jacobian_backend = Dict("sparse" => ADNLPModels.SparseADJacobian)
get_backend_list(::Val{:optimized}) = keys(benchmarked_jacobian_backend)
get_backend(::Val{:optimized}, b::String) = benchmarked_jacobian_backend[b]

problem_sets = Dict("scalable" => scalable_cons_problems)
nscal = 1000

name_backend = "jacobian_backend"
fun = jac_coord
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
          verbose_subbenchmark && @info " $(pb): $T with $n vars and $m cons"
          SUITE["$(fun)"][f][T][s][b][pb] = @benchmarkable $fun(nlp, get_x0(nlp)) setup =
            (nlp = set_adnlp($pb, $(name_backend), $backend, $nscal, $T))
        end
      end
    end
  end
end
