using LinearAlgebra, SparseArrays, Test
using SparseMatrixColorings
using ADNLPModels, ManualNLPModels, NLPModels, NLPModelsModifiers, NLPModelsTest
using ADNLPModels:
  gradient, gradient!, jacobian, hessian, Jprod!, Jtprod!, directional_second_derivative, Hvprod!

# Automatically loads the code for Enzyme with Requires
import Enzyme

EnzymeReverseAD() = ADNLPModels.ADModelBackend(
  ADNLPModels.EnzymeReverseADGradient(),
  ADNLPModels.EnzymeReverseADHvprod(zeros(1)),
  ADNLPModels.EnzymeReverseADJprod(zeros(1)),
  ADNLPModels.EnzymeReverseADJtprod(zeros(1)),
  ADNLPModels.EnzymeReverseADJacobian(),
  ADNLPModels.EnzymeReverseADHessian(zeros(1), zeros(1)),
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
    Jprod!(adbackend.jprod_backend, [1.0], sum!, [1.0], [1.0], Val(:c))
    Jtprod!(adbackend.jtprod_backend, [1.0], mysum!, [1.0], [1.0], Val(:c))
  end
end

test_autodiff_backend_error()

include("sparse_jacobian.jl")
include("sparse_jacobian_nls.jl")
include("sparse_hessian.jl")
include("sparse_hessian_nls.jl")

list_sparse_jac_backend = ((ADNLPModels.SparseEnzymeADJacobian, Dict()),)

@testset "Sparse Jacobian" begin
  for (backend, kw) in list_sparse_jac_backend
    sparse_jacobian(backend, kw)
    sparse_jacobian_nls(backend, kw)
  end
end

list_sparse_hess_backend = (
  (
    ADNLPModels.SparseEnzymeADHessian,
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:direct}()),
  ),
  (
    ADNLPModels.SparseEnzymeADHessian,
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:substitution}()),
  ),
)

@testset "Sparse Hessian" begin
  for (backend, kw) in list_sparse_hess_backend
    sparse_hessian(backend, kw)
    sparse_hessian_nls(backend, kw)
  end
end

for problem in NLPModelsTest.nlp_problems ∪ ["GENROSE"]
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in NLPModelsTest.nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end

include("utils.jl")
include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")

@testset "Basic NLP tests using $backend " for backend in (:enzyme,)
  test_autodiff_model("$backend", backend = backend)
end

@testset "Checking NLPModelsTest (NLP) tests with $backend" for backend in (:enzyme,)
  nlp_nlpmodelstest(backend)
end

@testset "Basic NLS tests using $backend " for backend in (:enzyme,)
  autodiff_nls_test("$backend", backend = backend)
end

@testset "Checking NLPModelsTest (NLS) tests with $backend" for backend in (:enzyme,)
  nls_nlpmodelstest(backend)
end
