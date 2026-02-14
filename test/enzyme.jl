using LinearAlgebra, SparseArrays, Test
using JET
using SparseMatrixColorings
using ADNLPModels, ManualNLPModels, NLPModels, NLPModelsModifiers, NLPModelsTest
using ADNLPModels:
  gradient, gradient!, jacobian, hessian, Jprod!, Jtprod!, directional_second_derivative, Hvprod!

# Automatically loads the code for Enzyme with Requires
import Enzyme

# Dummy constraint and Lagrangian functions for constructing an EnzymeReverseADHvprod
# with no constraints.  These are only used by EnzymeReverseAD() below, which builds
# a minimal ADModelBackend for the low-level smoke tests (gradient, jacobian, …).
# Real models get their own ℓ constructed from f and c! at model-creation time.
_noop_c!(y, x) = nothing
_noop_ℓ(x, y, obj_weight, cx) = zero(eltype(x))

# Construct an EnzymeReverseADHvprod with pre-allocated buffers of size `n` and
# no constraints (ncon = 0).  Fields: grad, hvbuf, xbuf, vbuf, cx, ybuf, f, c!, ℓ, ncon.
function _make_enzyme_hvprod(n)
  ADNLPModels.EnzymeReverseADHvprod(
    zeros(n),
    zeros(n),
    zeros(n),
    zeros(n),
    zeros(0),
    zeros(0),
    identity,
    _noop_c!,
    _noop_ℓ,
    0,
  )
end

EnzymeReverseAD() = ADNLPModels.ADModelBackend(
  ADNLPModels.EnzymeReverseADGradient(),
  _make_enzyme_hvprod(1),
  ADNLPModels.EnzymeReverseADJprod(zeros(1), zeros(1), zeros(1), zeros(1)),
  ADNLPModels.EnzymeReverseADJtprod(zeros(1), zeros(1), zeros(1), zeros(1)),
  ADNLPModels.EnzymeReverseADJacobian(),
  ADNLPModels.EnzymeReverseADHessian(zeros(1), zeros(1), identity),
  _make_enzyme_hvprod(1),
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

@testset "Basic NLP tests using enzyme" begin
  test_autodiff_model("enzyme", backend = :enzyme)
end

@testset "Basic NLS tests using enzyme" begin
  autodiff_nls_test("enzyme", backend = :enzyme)
end

@testset "Checking NLPModelsTest (NLP) tests with enzyme" begin
  nlp_nlpmodelstest(:enzyme)
end

@testset "Checking NLPModelsTest (NLS) tests with enzyme" begin
  nls_nlpmodelstest(:enzyme)
end
