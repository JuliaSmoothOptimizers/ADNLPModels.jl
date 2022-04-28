using ADNLPModels, LinearAlgebra, NLPModels, NLPModelsModifiers, NLPModelsTest, Test
using ADNLPModels:
  gradient, gradient!, jacobian, hessian, Jprod, Jtprod, directional_second_derivative, Hvprod

for problem in NLPModelsTest.nlp_problems ∪ ["GENROSE"]
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in NLPModelsTest.nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end

ForwardDiffAD(nvar, f, ncon = 0) = ADNLPModels.ADModelBackend(nvar, f, ncon)
ReverseDiffAD(nvar, f, ncon = 0) = ADNLPModels.ADModelBackend(
  nvar,
  f,
  ncon,
  gradient_backend = ADNLPModels.ReverseDiffADGradient,
  hprod_backend = ADNLPModels.ReverseDiffADHvprod,
  jprod_backend = ADNLPModels.ReverseDiffADJprod,
  jtprod_backend = ADNLPModels.ReverseDiffADJtprod,
  jacobian_backend = ADNLPModels.ReverseDiffADJacobian,
  hessian_backend = ADNLPModels.ReverseDiffADHessian,
)
ZygoteAD() = ADNLPModels.ADModelBackend(
  gradient_backend = ADNLPModels.ZygoteADGradient,
  jprod_backend = ADNLPModels.ZygoteADJprod,
  jtprod_backend = ADNLPModels.ZygoteADJtprod,
  jacobian_backend = ADNLPModels.ZygoteADJacobian,
  hessian_backend = ADNLPModels.ZygoteADHessian,
)

include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")
