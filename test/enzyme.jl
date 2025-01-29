using LinearAlgebra, SparseArrays, Test
using SparseMatrixColorings
using ADNLPModels, ManualNLPModels, NLPModels, NLPModelsModifiers, NLPModelsTest
using ADNLPModels:
  gradient, gradient!, jacobian, hessian, Jprod!, Jtprod!, directional_second_derivative, Hvprod!

# Automatically loads the code for Enzyme with Requires
import Enzyme

EnzymeReverseAD() = ADNLPModels.ADModelBackend(
  ADNLPModels.EnzymeReverseADGradient(),
  ADNLPModels.EnzymeReverseADHvprod(zeros(1), identity),
  ADNLPModels.EnzymeReverseADJprod(zeros(1)),
  ADNLPModels.EnzymeReverseADJtprod(zeros(1)),
  ADNLPModels.EnzymeReverseADJacobian(),
  ADNLPModels.EnzymeReverseADHessian(zeros(1), zeros(1), identity),
  ADNLPModels.EnzymeReverseADHvprod(zeros(1), identity),
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
  @testset "Enzyme basic operations - $backend" for backend in [:EnzymeReverseAD]
    adbackend = eval(backend)()
    gradient(adbackend.gradient_backend, sum, [1.0])
    gradient!(adbackend.gradient_backend, [1.0], sum, [1.0])
    jacobian(adbackend.jacobian_backend, sum, [1.0])
    hessian(adbackend.hessian_backend, sum, [1.0])
    Jprod!(adbackend.jprod_backend, [1.0], sum!, [1.0], [1.0], Val(:c))
    Jtprod!(adbackend.jtprod_backend, [1.0], mysum!, [1.0], [1.0], Val(:c))
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

const test_enzyme = true

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
    "star coloring",
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:direct}()),
  ),
  (
    ADNLPModels.SparseEnzymeADHessian,
    "acyclic coloring",
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:substitution}()),
  ),
)

@testset "Sparse Hessian" begin
  for (backend, info, kw) in list_sparse_hess_backend
    sparse_hessian(backend, info, kw)
    sparse_hessian_nls(backend, info, kw)
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

@testset "Basic NLP tests using enzyme_backend" begin
  test_autodiff_model("enzyme_backend", backend = :enzyme_backend)
end

@testset "Basic NLS tests using enzyme_backend" begin
  autodiff_nls_test("enzyme_backend", backend = :enzyme_backend)
end

@testset "Checking NLPModelsTest (NLP) tests with enzyme_backend" begin
  nlp_nlpmodelstest(:enzyme_backend)
end

@testset "Checking NLPModelsTest (NLS) tests with enzyme_backend" begin
  nls_nlpmodelstest(:enzyme_backend)
end
