using LinearAlgebra, SparseArrays, Test
using ADNLPModels, ManualNLPModels, NLPModels, NLPModelsModifiers, NLPModelsTest
using ADNLPModels:
  gradient, gradient!, jacobian, hessian, Jprod!, Jtprod!, directional_second_derivative, Hvprod!

for problem in NLPModelsTest.nlp_problems ∪ ["GENROSE"]
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in NLPModelsTest.nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end

ZygoteAD() = ADNLPModels.ADModelBackend(
  ADNLPModels.ZygoteADGradient(),
  ADNLPModels.GenericForwardDiffADHvprod(),
  ADNLPModels.ZygoteADJprod(),
  ADNLPModels.ZygoteADJtprod(),
  ADNLPModels.ZygoteADJacobian(0),
  ADNLPModels.ZygoteADHessian(0),
  ADNLPModels.ForwardDiffADGHjvprod(),
  ADNLPModels.EmptyADbackend(),
  ADNLPModels.EmptyADbackend(),
  ADNLPModels.EmptyADbackend(),
  ADNLPModels.EmptyADbackend(),
  ADNLPModels.EmptyADbackend(),
)

function test_autodiff_backend_error()
  @testset "Error without loading package - $backend" for backend in [:ZygoteAD]
    adbackend = eval(backend)()
    @test_throws ArgumentError gradient(adbackend.gradient_backend, sum, [1.0])
    @test_throws ArgumentError gradient!(adbackend.gradient_backend, [1.0], sum, [1.0])
    @test_throws ArgumentError jacobian(adbackend.jacobian_backend, identity, [1.0])
    @test_throws ArgumentError hessian(adbackend.hessian_backend, sum, [1.0])
    @test_throws ArgumentError Jprod!(
      adbackend.jprod_backend,
      [1.0],
      [1.0],
      identity,
      [1.0],
      Val(:c),
    )
    @test_throws ArgumentError Jtprod!(
      adbackend.jtprod_backend,
      [1.0],
      [1.0],
      identity,
      [1.0],
      Val(:c),
    )
  end
end

# Test the argument error without loading the packages
test_autodiff_backend_error()

# Additional backends used for tests
push!(
  ADNLPModels.predefined_backend,
  :zygote_backend => Dict(
    :gradient_backend => ADNLPModels.ZygoteADGradient,
    :jprod_backend => ADNLPModels.ZygoteADJprod,
    :jtprod_backend => ADNLPModels.ZygoteADJtprod,
    :hprod_backend => ADNLPModels.ForwardDiffADHvprod,
    :jacobian_backend => ADNLPModels.ZygoteADJacobian,
    :hessian_backend => ADNLPModels.ZygoteADHessian,
    :ghjvprod_backend => ADNLPModels.ForwardDiffADGHjvprod,
    :jprod_residual_backend => ADNLPModels.ZygoteADJprod,
    :jtprod_residual_backend => ADNLPModels.ZygoteADJtprod,
    :hprod_residual_backend => ADNLPModels.ForwardDiffADHvprod,
    :jacobian_residual_backend => ADNLPModels.ZygoteADJacobian,
    :hessian_residual_backend => ADNLPModels.ZygoteADHessian,
  ),
)

# Automatically loads the code for Zygote with Requires
import Zygote

include("utils.jl")
include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")
