#=
INTRODUCTION OF THIS BENCHMARK:

We test here the `hessian_backend` for ADNLPModels with different backends:
  - ADNLPModels.SparseADHessian;
  - ADNLPModels.SparseADHessian with Symbolics for sparsity detection.
=#
using ForwardDiff, SparseConnectivityTracer, SparseMatrixColorings, Symbolics

include("additional_backends.jl")

data_types = [Float64]

benchmark_list = [:optimized]

benchmarked_hess_coloring_backend = Dict(
  "sparse" => ADNLPModels.SparseADHessian,
  "sparse_symbolics" =>
    (nvar, f, ncon, c!; kwargs...) -> ADNLPModels.SparseADHessian(
      nvar,
      f,
      ncon,
      c!;
      detector = SymbolicsSparsityDetector(),
      kwargs...,
    ),
  # add ColPack?
)
get_backend_list(::Val{:optimized}) = keys(benchmarked_hess_coloring_backend)
get_backend(::Val{:optimized}, b::String) = benchmarked_hess_coloring_backend[b]

problem_sets = Dict("scalable" => scalable_cons_problems)
nscal = 1000

name_backend = "hessian_backend"
fun = :hessian_backend
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
          if m > 5 * nscal
            continue
          end
          verbose_subbenchmark && @info " $(pb): $T with $n vars and $m cons"
          SUITE["$(fun)"][f][T][s][b][pb] =
            @benchmarkable set_adnlp($pb, $(name_backend), $backend, $nscal, $T)
        end
      end
    end
  end
end
