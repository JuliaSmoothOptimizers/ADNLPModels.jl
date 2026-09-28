# Regression test for issue #383: the ReverseDiff-based backends record their tape by
# *executing* the model function, so the tape input must be a well-defined point.
#
# When it was allocated with `undef`, the primal parts held whatever the allocator handed
# back — legally including non-finite values, e.g. when a just-freed `-Inf` bounds array is
# recycled. Domain-restricted primitives (`cos` via `Base.sincos`, `sqrt`, `log`, ...) then
# threw a `DomainError` during construction, so the backend failed to build at all.
#
# `poison` recreates that condition deterministically by filling the small-object pool with
# the `-Inf` bit pattern before each construction.

@testset "Tape inputs are seeded from x0 (issue #383)" begin
  f(x) = sum(x)
  c!(cx, x) = (cx[1] = cos(x[1]) - 1; cx)
  nvar, ncon = 4, 1
  Hpat = sparse(1:nvar, 1:nvar, trues(nvar), nvar, nvar)

  ψ = (x, u) -> (tmp = similar(x, length(u)); c!(tmp, x); dot(tmp, u))
  tagψ = ForwardDiff.Tag{typeof(ψ), Float64}
  neginf_dual = ForwardDiff.Dual{tagψ}(-Inf, -Inf)

  function poison()
    for _ = 1:5000
      b = fill(neginf_dual, nvar)
      Base.donotdelete(b)
    end
    GC.gc()
    GC.gc()
    return nothing
  end

  # Both backends used to record on uninitialised memory. `SparseADHessian` and the
  # ForwardDiff backends never did, and are kept here as controls.
  builders = (
    "SparseReverseADHessian" => () -> ADNLPModels.SparseReverseADHessian(nvar, f, ncon, c!, Hpat),
    "ReverseDiffADHvprod" => () -> ADNLPModels.ReverseDiffADHvprod(nvar, f, ncon, c!),
    "SparseADHessian" => () -> ADNLPModels.SparseADHessian(nvar, f, ncon, c!, Hpat),
    "ForwardDiffADHvprod" => () -> ADNLPModels.ForwardDiffADHvprod(nvar, f, ncon, c!),
  )

  for (name, build) in builders
    @testset "$name" begin
      for _ = 1:10
        poison()
        @test build() isa ADNLPModels.ADBackend
      end
    end
  end

  # End to end: a model whose constraint is only defined on finite inputs must build and
  # evaluate under the `:optimized` preset, which selects `SparseReverseADHessian`.
  @testset "ADNLPModel with :optimized" begin
    poison()
    x0 = collect(range(0.1, 0.4, length = nvar))
    nlp = ADNLPModel!(f, x0, c!, [0.0], [0.0]; backend = :optimized)
    @test grad(nlp, x0) ≈ ones(nvar)
    H = hess(nlp, x0, [1.0])
    @test H[1, 1] ≈ -cos(x0[1])
  end
end
