using LinearAlgebra, SparseArrays, Test
using SparseMatrixColorings
using JET
using ADNLPModels, ManualNLPModels, NLPModels, NLPModelsModifiers, NLPModelsTest
using ADNLPModels:
  gradient, gradient!, jacobian, hessian, Jprod!, Jtprod!, directional_second_derivative, Hvprod!

@testset "Test sparsity pattern of Jacobian and Hessian" begin
  f(x) = sum(x .^ 2)
  c(x) = x
  c!(cx, x) = copyto!(cx, x)
  nvar, ncon = 2, 2
  x0 = ones(nvar)
  cx = rand(ncon)
  S = ADNLPModels.compute_jacobian_sparsity(c, x0)
  @test S == I
  S = ADNLPModels.compute_jacobian_sparsity(c!, cx, x0)
  @test S == I
  S = ADNLPModels.compute_hessian_sparsity(f, nvar, c!, ncon)
  @test S == I
end

@testset "Test using a NLPModel instead of AD-backend" begin
  include("manual.jl")
end

include("sparse_jacobian.jl")
include("sparse_jacobian_nls.jl")
include("sparse_hessian.jl")
include("sparse_hessian_nls.jl")

list_sparse_jac_backend =
  ((ADNLPModels.SparseADJacobian, Dict()), (ADNLPModels.ForwardDiffADJacobian, Dict()))

@testset "Sparse Jacobian" begin
  for (backend, kw) in list_sparse_jac_backend
    sparse_jacobian(backend, kw)
    sparse_jacobian_nls(backend, kw)
  end
end

list_sparse_hess_backend = (
  (ADNLPModels.SparseADHessian, Dict(:coloring_algorithm => GreedyColoringAlgorithm{:direct}())),
  (
    ADNLPModels.SparseADHessian,
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:substitution}()),
  ),
  (
    ADNLPModels.SparseReverseADHessian,
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:direct}()),
  ),
  (
    ADNLPModels.SparseReverseADHessian,
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:substitution}()),
  ),
  (ADNLPModels.ForwardDiffADHessian, Dict()),
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
include("nlp/nlpmodelstest.jl")
include("nls/basic.jl")
include("nls/nlpmodelstest.jl")

@testset "Basic NLP tests using $backend " for backend in keys(ADNLPModels.predefined_backend)
  (backend == :zygote) && continue
  (backend == :enzyme) && continue
  test_autodiff_model("$backend", backend = backend)
end

@testset "Checking NLPModelsTest (NLP) tests with $backend" for backend in
                                                                keys(ADNLPModels.predefined_backend)
  (backend == :zygote) && continue
  (backend == :enzyme) && continue
  nlp_nlpmodelstest(backend)
end

@testset "Basic NLS tests using $backend " for backend in keys(ADNLPModels.predefined_backend)
  (backend == :zygote) && continue
  (backend == :enzyme) && continue
  autodiff_nls_test("$backend", backend = backend)
end

@testset "Checking NLPModelsTest (NLS) tests with $backend" for backend in
                                                                keys(ADNLPModels.predefined_backend)
  (backend == :zygote) && continue
  (backend == :enzyme) && continue
  nls_nlpmodelstest(backend)
end
