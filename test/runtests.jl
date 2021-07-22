using ADNLPModels, LinearAlgebra, NLPModels, NLPModelsModifiers, NLPModelsTest, Test
using ADNLPModels:
  ForwardDiffAD,
  ZygoteAD,
  ReverseDiffAD,
  gradient,
  gradient!,
  jacobian,
  hessian,
  Jprod,
  Jtprod,
  directional_second_derivative,
  Hvprod

for problem in NLPModelsTest.nlp_problems ∪ ["GENROSE"]
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in NLPModelsTest.nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end

function backends()
  x = get(ENV, "ADBACKEND", nothing)
  if x === nothing
    return (:ForwardDiffAD, :ZygoteAD, :ReverseDiffAD)
  elseif get(ENV, "GITHUB_REPOSITORY", nothing) == "JuliaSmoothOptimizers/ADNLPModels.jl"
    return [Symbol(x)]
  else #
    return [:ForwardDiffAD]
  end
end

include("nlp/basic.jl")
include("nls/basic.jl")

if get(ENV, "CI", "false") == "true" && get(ENV, "GITHUB_REPOSITORY", nothing) == "JuliaSmoothOptimizers/ADNLPModels.jl"
  if get(ENV, "PROBLEMTYPE", nothing) == "NLP"
    include("nlp/nlpmodelstest.jl")
  else
    include("nls/nlpmodelstest.jl")
  end
else
  include("nlp/nlpmodelstest.jl")
  include("nls/nlpmodelstest.jl")
end
