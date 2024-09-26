using LinearAlgebra, SparseArrays, Test
using SparseMatrixColorings
using ADNLPModels, ManualNLPModels, NLPModels, NLPModelsModifiers, NLPModelsTest
using ADNLPModels:
  gradient, gradient!, jacobian, hessian, Jprod!, Jtprod!, directional_second_derivative, Hvprod!
import DifferentiationInterface: MissingBackendError

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

@testset "Basic Jacobian derivative test" begin
  include("sparse_jacobian.jl")
  include("sparse_jacobian_nls.jl")
end

@testset "Basic Hessian derivative test" begin
  include("sparse_hessian.jl")
  include("sparse_hessian_nls.jl")
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

@testset "Error without loading package - $package" for package in
  [:Enzyme, :Zygote, :Mooncake, :Diffractor, :Tracker, :Symbolics, :ChainRules,
   :FastDifferentiation, :FiniteDiff, :FiniteDifferences, :PolyesterForwardDiff]
  adbackend = ADNLPModels.predefined_backend[package]
  @test_throws MissingBackendError gradient(adbackend[:gradient_backend](1, x -> sum(x)), sum, [1.0])
  @test_throws MissingBackendError gradient!(adbackend[:gradient_backend](1, x -> sum(x)), [1.0], sum, [1.0])
end

include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")
