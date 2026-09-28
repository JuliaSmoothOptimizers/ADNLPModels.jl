"""
Shared helpers for the `DifferentiationInterface.jl` (DI) backends.

ADNLPModels uses DI as a *dense* AD kernel only: sparsity pattern detection
(`SparseConnectivityTracer`) and coloring/decompression (`SparseMatrixColorings`) are
handled by ADNLPModels itself, because the sparse Jacobians and Hessians are stored in a
single triangle, which DI's `AutoSparse` cannot produce.
"""

# The `AbstractADType`s wrapped by the backends. `NC` variants ("no compile") are used
# wherever the tape cannot be reused: either the element type varies at run time (the
# `Generic*` backends) or the differentiated function changes between calls.
const FDBackend = AutoForwardDiff()
const RDBackend = AutoReverseDiff(compile = Val(true))
const RDBackendNC = AutoReverseDiff()

# Forward-over-reverse: the cheapest second-order composition for Hessian-vector products.
const FoRBackend = SecondOrder(AutoForwardDiff(), AutoReverseDiff(compile = Val(true)))
const FoRBackendNC = SecondOrder(AutoForwardDiff(), AutoReverseDiff())

"""
    LagrangianFunction(f, c!, ncon)

Callable computing `ℓ(x, cx, y, ob) = ob * f(x) + dot(c(x), y)`, with `cx` a scratch buffer
for the in-place constraints `c!`.

`y` and `ob` are passed to DI as `Constant` contexts and `cx` as a `Cache`, so that a single
preparation covers every Hessian call: the Lagrangian (`y`, `ob`), the objective alone
(`y = 0`), and the `j`-th constraint or residual (`y = eⱼ`, `ob = 0`).

This replaces the augmented function over `z = (cx, x, y, ob)` of length `nvar + 2 * ncon + 1`
that ADNLPModels used to build by hand: differentiating with respect to `x` alone is both
simpler and cheaper.

A named struct is used rather than a closure because DI records `typeof(f)` in the
preparation signature, and a closure's type would vary with the type of the captured `y`.
"""
struct LagrangianFunction{F, C}
  f::F
  c!::C
  ncon::Int
end

function (ℓ::LagrangianFunction)(x, cx, y, ob)
  if ℓ.ncon > 0
    ℓ.c!(cx, x)
    return ob * ℓ.f(x) + dot(cx, y)
  else
    return ob * ℓ.f(x)
  end
end

"""
    DotConstraints(c!)

Callable computing `ψ(x, cx, u) = dot(c(x), u)`, whose gradient with respect to `x` is the
transposed Jacobian-vector product `J(x)ᵀu`.

Used instead of `DI.pullback` for forward-mode backends: DI implements the pullback of a
forward-mode backend as one pushforward per component of `x`, which costs `nvar` evaluations.
"""
struct DotConstraints{C}
  c!::C
end

(ψ::DotConstraints)(x, cx, u) = (ψ.c!(cx, x); dot(cx, u))

"""
    seeded_duals(tag, x0, nvar)

Allocate a `Vector{ForwardDiff.Dual{tag, T, 1}}` of length `nvar` whose primal parts are
initialised from `x0` and whose partials are zero.

`ReverseDiff.GradientTape` *executes* the function while recording the tape, so its input
must be a well-defined point. Allocating it with `undef` leaves the primal parts holding
whatever the allocator hands back, which may legally be non-finite (a recycled `-Inf` bounds
array, say). Domain-restricted primitives such as `cos`, `sqrt` or `log` then throw during
tape construction, so the backend fails to build at all. See issue #383.

Recording at a fixed point also makes tape compilation deterministic.
"""
function seeded_duals(::Type{tag}, x0::AbstractVector{T}, nvar::Integer) where {tag, T}
  z = Vector{ForwardDiff.Dual{tag, T, 1}}(undef, nvar)
  @inbounds for i = 1:nvar
    z[i] = ForwardDiff.Dual{tag}(x0[i], zero(T))
  end
  return z
end
