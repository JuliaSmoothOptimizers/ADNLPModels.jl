using ADNLPModels, LinearAlgebra, NLPModels, NLPModelsModifiers, NLPModelsTest, SparseArrays, Test
using ADNLPModels:
  gradient, gradient!, jacobian, hessian, Jprod!, Jtprod!, directional_second_derivative, Hvprod!

using SparseDiffTools
@testset "Basic Jacobian derivative test" begin
  include("sparse_jacobian.jl")
  include("sparse_jacobian_nls.jl")
end

@testset "Basic Hessian derivative test" begin
  include("sparse_hessian.jl")
end

for problem in NLPModelsTest.nlp_problems ∪ ["GENROSE"]
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in NLPModelsTest.nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end

OptimizedAD(nvar, f) = ADNLPModels.ADModelBackend(nvar, f)
ForwardDiffAD(nvar, f) = ADNLPModels.ADModelBackend(
  nvar,
  f,
  gradient_backend = ADNLPModels.GenericForwardDiffADGradient,
  hprod_backend = ADNLPModels.GenericForwardDiffADHvprod,
  jprod_backend = ADNLPModels.GenericForwardDiffADJprod,
  jtprod_backend = ADNLPModels.ForwardDiffADJtprod,
  jacobian_backend = ADNLPModels.ForwardDiffADJacobian,
  hessian_backend = ADNLPModels.ForwardDiffADHessian,
)
ReverseDiffAD(nvar, f) = ADNLPModels.ADModelBackend(
  nvar,
  f,
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

function test_getter_setter(nlp)
  @test get_adbackend(nlp) == nlp.adbackend
  if typeof(nlp) <: ADNLPModel
    set_adbackend!(nlp, ReverseDiffAD(nlp.meta.nvar, nlp.f))
  elseif typeof(nlp) <: ADNLSModel
    function F(x; nequ = nlp.nls_meta.nequ)
      Fx = similar(x, nequ)
      nlp.F!(Fx, x)
      return Fx
    end
    set_adbackend!(nlp, ReverseDiffAD(nlp.meta.nvar, x -> sum(F(x) .^ 2)))
  end
  @test typeof(get_adbackend(nlp).gradient_backend) <: ADNLPModels.ReverseDiffADGradient
  @test typeof(get_adbackend(nlp).hprod_backend) <: ADNLPModels.ReverseDiffADHvprod
  @test typeof(get_adbackend(nlp).hessian_backend) <: ADNLPModels.ReverseDiffADHessian
  set_adbackend!(
    nlp,
    gradient_backend = ADNLPModels.ForwardDiffADGradient,
    jtprod_backend = ADNLPModels.GenericForwardDiffADJtprod(),
  )
  @test typeof(get_adbackend(nlp).gradient_backend) <: ADNLPModels.ForwardDiffADGradient
  @test typeof(get_adbackend(nlp).hprod_backend) <: ADNLPModels.ReverseDiffADHvprod
  @test typeof(get_adbackend(nlp).jtprod_backend) <: ADNLPModels.GenericForwardDiffADJtprod
  @test typeof(get_adbackend(nlp).hessian_backend) <: ADNLPModels.ReverseDiffADHessian
end

include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")
