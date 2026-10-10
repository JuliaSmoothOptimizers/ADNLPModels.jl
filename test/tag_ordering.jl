# Regression test: a constraint with a component that does not depend on `x` (e.g. the
# dynamics `ẋ₃ = 0` of an optimal control problem) used to fail with
# "Cannot determine ordering of Dual tags".
#
# The constant is recorded as an untracked value, so ReverseDiff differentiates the binary
# operation that consumes it with its own nested `ForwardDiff.Dual`, whose tag must be ordered
# after ours. ForwardDiff orders tags by a counter that is only assigned when the tag is built
# with `ForwardDiff.Tag(f, T)`; spelling the type `ForwardDiff.Tag{typeof(f), T}` skipped it,
# so our tag could be numbered after ReverseDiff's and the ordering was reversed.

@testset "Dual tag ordering with constant constraint components" begin
  f(x) = x[1]^2
  function c!(cx, x)
    d = [x[2] * x[3], x[1], 0]
    for i = 1:3
      cx[i] = x[i] - d[i] / 2
    end
    return cx
  end
  x, y, v = ones(3), ones(3), [1.0, 2.0, 3.0]
  H = [2.0 0.0 0.0; 0.0 0.0 -0.5; 0.0 -0.5 0.0]

  nlp =
    ADNLPModel!(f, x, c!, zeros(3), zeros(3), hessian_backend = ADNLPModels.SparseReverseADHessian)
  @test hess(nlp, x, y) ≈ H

  nlp = ADNLPModel!(f, x, c!, zeros(3), zeros(3), hprod_backend = ADNLPModels.ReverseDiffADHvprod)
  @test hprod(nlp, x, y, v) ≈ H * v
end
