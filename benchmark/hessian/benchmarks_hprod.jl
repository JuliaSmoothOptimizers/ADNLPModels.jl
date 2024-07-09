#=
INTRODUCTION OF THIS BENCHMARK:

We test here the function `hprod!` for ADNLPModels with different backends:
  - ADNLPModels.ForwardDiffADHvprod
  - ADNLPModels.ReverseDiffADHvprod
=#
using ForwardDiff, ReverseDiff

include("additional_backends.jl")

data_types = [Float32, Float64]

benchmark_list = [:optimized]

benchmarked_hprod_backend =
  Dict("forward" => ADNLPModels.ForwardDiffADHvprod, "reverse" => ADNLPModels.ReverseDiffADHvprod)
get_backend_list(::Val{:optimized}) = keys(benchmarked_hprod_backend)
get_backend(::Val{:optimized}, b::String) = benchmarked_hprod_backend[b]

problem_sets = Dict("scalable" => scalable_problems)
nscal = 1000

name_backend = "hprod_backend"
fun = hprod!
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
          verbose_subbenchmark && @info " $(pb): $T with $n vars"
          v = [sin(T(i) / 10) for i = 1:n]
          Hv = Vector{T}(undef, n)
          SUITE["$(fun)"][f][T][s][b][pb] = @benchmarkable $fun(nlp, get_x0(nlp), $v, $Hv) setup =
            (nlp = set_adnlp($pb, $(name_backend), $backend, $nscal, $T))
        end
      end
    end
  end
end
