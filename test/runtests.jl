using LinearAlgebra, SparseArrays, Test
using SparseMatrixColorings
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
  (
    ADNLPModels.SparseADHessian,
    "star coloring with postprocessing",
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:direct}(postprocessing = true)),
  ),
  (
    ADNLPModels.SparseADHessian,
    "star coloring without postprocessing",
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:direct}(postprocessing = false)),
  ),
  (
    ADNLPModels.SparseADHessian,
    "acyclic coloring with postprocessing",
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:substitution}(postprocessing = true)),
  ),
  (
    ADNLPModels.SparseADHessian,
    "acyclic coloring without postprocessing",
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:substitution}(postprocessing = false)),
  ),
  (
    ADNLPModels.SparseReverseADHessian,
    "star coloring with postprocessing",
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:direct}(postprocessing = true)),
  ),
  (
    ADNLPModels.SparseReverseADHessian,
    "star coloring without postprocessing",
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:direct}(postprocessing = false)),
  ),
  (
    ADNLPModels.SparseReverseADHessian,
    "acyclic coloring with postprocessing",
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:substitution}(postprocessing = true)),
  ),
  (
    ADNLPModels.SparseReverseADHessian,
    "acyclic coloring without postprocessing",
    Dict(:coloring_algorithm => GreedyColoringAlgorithm{:substitution}(postprocessing = false)),
  ),
  (ADNLPModels.ForwardDiffADHessian, "default", Dict()),
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
include("nlp/nlpmodelstest.jl")
include("nls/basic.jl")
include("nls/nlpmodelstest.jl")

@testset "Basic NLP tests using $backend " for backend in keys(ADNLPModels.predefined_backend)
  (backend == :enzyme) && continue
  test_autodiff_model("$backend", backend = backend)
end

@testset "Checking NLPModelsTest (NLP) tests with $backend" for backend in
                                                                keys(ADNLPModels.predefined_backend)
  (backend == :enzyme) && continue
  nlp_nlpmodelstest(backend)
end

@testset "Basic NLS tests using $backend " for backend in keys(ADNLPModels.predefined_backend)
  (backend == :enzyme) && continue
  autodiff_nls_test("$backend", backend = backend)
end

@testset "Checking NLPModelsTest (NLS) tests with $backend" for backend in
                                                                keys(ADNLPModels.predefined_backend)
  (backend == :enzyme) && continue
  nls_nlpmodelstest(backend)
end
