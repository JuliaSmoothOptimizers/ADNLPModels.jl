using CUDA, LinearAlgebra, SparseArrays, Test
using ADNLPModels, ManualNLPModels, NLPModels, NLPModelsModifiers, NLPModelsTest
using ADNLPModels:
  gradient, gradient!, jacobian, hessian, Jprod!, Jtprod!, directional_second_derivative, Hvprod!

@testset "Error without loading package for sparsity pattern" begin
  f(x) = sum(x)
  c!(cx, x) = begin
    cx .= 1
    return x
  end
  nvar, ncon = 2, 1
  x0 = ones(nvar)
  cx = rand(ncon)
  @test_throws ArgumentError ADNLPModels.compute_jacobian_sparsity(c!, cx, x0)
  @test_throws ArgumentError ADNLPModels.compute_hessian_sparsity(f, nvar, c!, ncon)
end

using SparseDiffTools, Symbolics

@testset "Test using a NLPModel instead of AD-backend" begin
  include("manual.jl")
end

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

# Additional backends used for tests
push!(
  ADNLPModels.predefined_backend,
  :zygote_backend => Dict(
    :gradient_backend => ADNLPModels.ZygoteADGradient,
    :hprod_backend => ADNLPModels.SDTForwardDiffADHvprod,
    :jprod_backend => ADNLPModels.ZygoteADJprod,
    :jtprod_backend => ADNLPModels.ZygoteADJtprod,
    :jacobian_backend => ADNLPModels.ZygoteADJacobian,
    :hessian_backend => ADNLPModels.ZygoteADHessian,
    :ghjvprod_backend => ADNLPModels.ForwardDiffADGHjvprod,
    :hprod_residual_backend => ADNLPModels.SDTForwardDiffADHvprod,
    :jprod_residual_backend => ADNLPModels.ZygoteADJprod,
    :jtprod_residual_backend => ADNLPModels.ZygoteADJtprod,
    :jacobian_residual_backend => ADNLPModels.ZygoteADJacobian,
    :hessian_residual_backend => ADNLPModels.ZygoteADHessian,
  ),
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

# Automatically loads the code for Zygote with Requires
import Zygote

include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")

if CUDA.functional()
  @testset "NLPModelsTest (NLP) tests - GPU multiple precision" begin
    include("gpu.jl")
  end
end
