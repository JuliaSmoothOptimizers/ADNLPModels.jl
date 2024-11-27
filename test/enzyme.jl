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

EnzymeReverseAD() = ADNLPModels.ADModelBackend(
  ADNLPModels.EnzymeReverseADGradient(),
  ADNLPModels.EnzymeReverseADHvprod(zeros(1)),
  ADNLPModels.EnzymeReverseADJprod(zeros(1)),
  ADNLPModels.EnzymeReverseADJtprod(zeros(1)),
  ADNLPModels.EnzymeReverseADJacobian(),
  ADNLPModels.EnzymeReverseADHessian(),
  ADNLPModels.EnzymeReverseADHvprod(zeros(1)),
  ADNLPModels.EmptyADbackend(),
  ADNLPModels.EmptyADbackend(),
  ADNLPModels.EmptyADbackend(),
  ADNLPModels.EmptyADbackend(),
  ADNLPModels.EmptyADbackend(),
)
function mysum!(y, x)
  sum!(y, x)
  return nothing
end
function test_autodiff_backend_error()
  @testset "Error without loading package - $backend" for backend in [:EnzymeReverseAD]
    adbackend = eval(backend)()
    # @test_throws ArgumentError gradient(adbackend.gradient_backend, sum, [1.0])
    # @test_throws ArgumentError gradient!(adbackend.gradient_backend, [1.0], sum, [1.0])
    # @test_throws ArgumentError jacobian(adbackend.jacobian_backend, identity, [1.0])
    # @test_throws ArgumentError hessian(adbackend.hessian_backend, sum, [1.0])
    # @test_throws ArgumentError Jprod!(
    #   adbackend.jprod_backend,
    #   [1.0],
    #   [1.0],
    #   identity,
    #   [1.0],
    #   Val(:c),
    # )
    # @test_throws ArgumentError Jtprod!(
    #   adbackend.jtprod_backend,
    #   [1.0],
    #   [1.0],
    #   identity,
    #   [1.0],
    #   Val(:c),
    # )
    gradient(adbackend.gradient_backend, sum, [1.0])
    gradient!(adbackend.gradient_backend, [1.0], sum, [1.0])
    jacobian(adbackend.jacobian_backend, sum, [1.0])
    hessian(adbackend.hessian_backend, sum, [1.0])
    Jprod!(
      adbackend.jprod_backend,
      [1.0],
      sum!,
      [1.0],
      [1.0],
      Val(:c),
    )
    Jtprod!(
      adbackend.jtprod_backend,
      [1.0],
      mysum!,
      [1.0],
      [1.0],
      Val(:c),
    )
  end
end

test_autodiff_backend_error()

push!(
  ADNLPModels.predefined_backend,
  :enzyme_backend => Dict(
    :gradient_backend => ADNLPModels.EnzymeReverseADGradient,
    :jprod_backend => ADNLPModels.EnzymeReverseADJprod,
    :jtprod_backend => ADNLPModels.EnzymeReverseADJtprod,
    :hprod_backend => ADNLPModels.EnzymeReverseADHvprod,
    :jacobian_backend => ADNLPModels.EnzymeReverseADJacobian,
    :hessian_backend => ADNLPModels.EnzymeReverseADHessian,
    :ghjvprod_backend => ADNLPModels.ForwardDiffADGHjvprod,
    :jprod_residual_backend => ADNLPModels.EnzymeReverseADJprod,
    :jtprod_residual_backend => ADNLPModels.EnzymeReverseADJtprod,
    :hprod_residual_backend => ADNLPModels.EnzymeReverseADHvprod,
    :jacobian_residual_backend => ADNLPModels.EnzymeReverseADJacobian,
    :hessian_residual_backend => ADNLPModels.EnzymeReverseADHessian,
  ),
)

include("utils.jl")
include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")

