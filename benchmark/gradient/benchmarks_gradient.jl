#=
INTRODUCTION OF THIS BENCHMARK:

We test here the function `grad!` for ADNLPModels with different backends:
  - ADNLPModels.ForwardDiffADGradient (use ForwardDiff.jl);
  - ADNLPModels.ReverseDiffADGradient (use ReverseDiff.jl);
  - DNLPModels.EnzymeADGradient (use Enzyme.jl);
  - ADNLPModels.ZygoteADGradient (use Zygote.jl).
=#
using ReverseDiff, Zygote, ForwardDiff, Enzyme

include("additional_backends.jl")

data_types = [Float32, Float64]

benchmark_list = [:optimized, :generic]

benchmarked_gradient_backend = Dict(
  "forward" => ADNLPModels.ForwardDiffADGradient,
  "reverse" => ADNLPModels.ReverseDiffADGradient,
  # "enzyme" => ADNLPModels.EnzymeADGradient,
)
get_backend_list(::Val{:optimized}) = keys(benchmarked_gradient_backend)
get_backend(::Val{:optimized}, b::String) = benchmarked_gradient_backend[b]

benchmarked_generic_gradient_backend = Dict(
  "forward" => ADNLPModels.GenericForwardDiffADGradient,
  "reverse" => ADNLPModels.GenericReverseDiffADGradient,
  #"zygote" => ADNLPModels.ZygoteADGradient, # ERROR: Mutating arrays is not supported
)
get_backend_list(::Val{:generic}) = keys(benchmarked_generic_gradient_backend)
get_backend(::Val{:generic}, b::String) = benchmarked_generic_gradient_backend[b]

problem_sets = Dict("scalable" => scalable_problems)
nscal = 1000

name_backend = "gradient_backend"
fun = grad!
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
          g = zeros(T, n)
          SUITE["$(fun)"][f][T][s][b][pb] = @benchmarkable $fun(nlp, get_x0(nlp), $g) setup =
            (nlp = set_adnlp($pb, $(name_backend), $(backend), $nscal, $T))
        end
      end
    end
  end
end
