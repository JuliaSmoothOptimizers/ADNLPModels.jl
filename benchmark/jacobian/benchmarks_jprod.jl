#=
INTRODUCTION OF THIS BENCHMARK:

We test here the function `jprod` for ADNLPModels with different backends:
  - ADNLPModels.ForwardDiffADJprod
  - ADNLPModels.ReverseDiffADJprod
=#
using ForwardDiff, ReverseDiff

include("additional_backends.jl")

data_types = [Float32, Float64]

benchmark_list = [:optimized]

benchmarked_jprod_backend =
  Dict("forward" => ADNLPModels.ForwardDiffADJprod, "reverse" => ADNLPModels.ReverseDiffADJprod)
get_backend_list(::Val{:optimized}) = keys(benchmarked_jprod_backend)
get_backend(::Val{:optimized}, b::String) = benchmarked_jprod_backend[b]

problem_sets = Dict("scalable" => scalable_cons_problems)
nscal = 1000

name_backend = "jprod_backend"
fun = jprod!
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
          @info " $(pb): $T with $n vars and $m cons"
          Jv = Vector{T}(undef, m)
          v = 10 * T[-(-1.0)^i for i = 1:n]
          SUITE["$(fun)"][f][T][s][b][pb] = @benchmarkable $fun(nlp, get_x0(nlp), $v, $Jv) setup =
            (nlp = set_adnlp($pb, $(name_backend), $backend, $nscal, $T))
        end
      end
    end
  end
end
