using ADNLPModels, LinearAlgebra, NLPModels, NLPModelsModifiers, NLPModelsTest, Test
using ADNLPModels:  ForwardDiffAD, ZygoteAD, ReverseDiffAD,
                    gradient, gradient!, jacobian, hessian, Jprod,
                    Jtprod, directional_second_derivative, Hvprod

for problem in NLPModelsTest.nlp_problems ∪ ["GENROSE"]
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in NLPModelsTest.nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end

include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")
