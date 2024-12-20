using LinearAlgebra, SparseArrays, Test
using SparseMatrixColorings
using ADNLPModels, ManualNLPModels, NLPModels, NLPModelsModifiers, NLPModelsTest
using ADNLPModels:
  gradient, gradient!, jacobian, hessian, Jprod!, Jtprod!, directional_second_derivative, Hvprod!

const test_enzyme = false

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

include("utils.jl")
include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")
