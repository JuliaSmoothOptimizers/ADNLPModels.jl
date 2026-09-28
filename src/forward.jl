struct GenericForwardDiffADGradient <: ADBackend end
GenericForwardDiffADGradient(args...; kwargs...) = GenericForwardDiffADGradient()
function gradient!(::GenericForwardDiffADGradient, g, f, x)
  return DI.gradient!(f, g, FDBackend, x)
end

struct ForwardDiffADGradient{B, P} <: ADBackend
  backend::B
  prep::P
end
function ForwardDiffADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  @assert nvar > 0
  @lencheck nvar x0
  prep = DI.prepare_gradient(f, FDBackend, x0; strict = Val(false))
  return ForwardDiffADGradient(FDBackend, prep)
end
function gradient!(b::ForwardDiffADGradient, g, f, x)
  return DI.gradient!(f, g, b.prep, b.backend, x)
end

struct ForwardDiffADJacobian <: ADBackend
  nnzj::Int
end
function ForwardDiffADJacobian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzj = nvar * ncon
  return ForwardDiffADJacobian(nnzj)
end
jacobian(::ForwardDiffADJacobian, f, x) = DI.jacobian(f, FDBackend, x)

struct ForwardDiffADHessian <: ADBackend
  nnzh::Int
end
function ForwardDiffADHessian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzh = nvar * (nvar + 1) / 2
  return ForwardDiffADHessian(nnzh)
end
hessian(::ForwardDiffADHessian, f, x) = DI.hessian(f, FDBackend, x)

struct GenericForwardDiffADJprod <: ADBackend end
function GenericForwardDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericForwardDiffADJprod()
end
function Jprod!(::GenericForwardDiffADJprod, Jv, f, x, v, ::Val)
  DI.pushforward!(f, (Jv,), FDBackend, x, (v,))
  return Jv
end

struct ForwardDiffADJprod{B, P, S} <: InPlaceADbackend
  backend::B
  prep::P
  cx::S
end

function ForwardDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  cx = similar(x0, ncon)
  v0 = fill!(similar(x0, nvar), zero(T))
  prep = DI.prepare_pushforward(c!, cx, FDBackend, x0, (v0,); strict = Val(false))
  return ForwardDiffADJprod(FDBackend, prep, cx)
end

function Jprod!(b::ForwardDiffADJprod, Jv, c!, x, v, ::Val)
  DI.pushforward!(c!, b.cx, (Jv,), b.prep, b.backend, x, (v,))
  return Jv
end

struct GenericForwardDiffADJtprod <: ADBackend end
function GenericForwardDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericForwardDiffADJtprod()
end
function Jtprod!(::GenericForwardDiffADJtprod, Jtv, f, x, v, ::Val)
  DI.gradient!(x -> dot(f(x), v), Jtv, FDBackend, x)
  return Jtv
end

struct ForwardDiffADJtprod{B, P, GT, S} <: InPlaceADbackend
  backend::B
  ψ::GT
  prep::P
  cx::S
end

function ForwardDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  ψ = DotConstraints(c!)
  cx = similar(x0, ncon)
  u0 = fill!(similar(x0, ncon), zero(T))
  prep = DI.prepare_gradient(ψ, FDBackend, x0, Cache(cx), Constant(u0); strict = Val(false))
  return ForwardDiffADJtprod(FDBackend, ψ, prep, cx)
end

function Jtprod!(b::ForwardDiffADJtprod, Jtv, c!, x, v, ::Val)
  DI.gradient!(b.ψ, Jtv, b.prep, b.backend, x, Cache(b.cx), Constant(v))
  return Jtv
end

struct GenericForwardDiffADHvprod <: ADBackend end
function GenericForwardDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericForwardDiffADHvprod()
end
function Hvprod!(::GenericForwardDiffADHvprod, Hv, x, v, f, args...)
  DI.hvp!(f, (Hv,), FDBackend, x, (v,))
  return Hv
end

struct ForwardDiffADHvprod{B, F, L, P1, P2, S} <: ADBackend
  backend::B
  f::F
  ℓ::L
  prep_obj::P1
  prep_lag::P2
  cx::S
  y::S
end

function ForwardDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::S = rand(nvar),
  kwargs...,
) where {S}
  T = eltype(S)
  ℓ = LagrangianFunction(f, c!, ncon)
  cx = similar(x0, ncon)
  y = fill!(similar(x0, ncon), zero(T))
  v0 = fill!(similar(x0, nvar), zero(T))

  prep_obj = DI.prepare_hvp(f, FDBackend, x0, (v0,); strict = Val(false))
  prep_lag = DI.prepare_hvp(
    ℓ,
    FDBackend,
    x0,
    (v0,),
    Cache(cx),
    Constant(y),
    Constant(one(T));
    strict = Val(false),
  )

  return ForwardDiffADHvprod(FDBackend, f, ℓ, prep_obj, prep_lag, cx, y)
end

# `y` is copied into the backend-owned buffer `b.y` so that the `Constant` context always has
# the same type as the one used at preparation, even when the caller passes a view.
function Hvprod!(
  b::ForwardDiffADHvprod,
  Hv,
  x::AbstractVector{T},
  v,
  ℓ,
  ::Val{:lag},
  y,
  obj_weight::Real = one(T),
) where {T}
  b.y .= y
  DI.hvp!(
    b.ℓ,
    (Hv,),
    b.prep_lag,
    b.backend,
    x,
    (v,),
    Cache(b.cx),
    Constant(b.y),
    Constant(T(obj_weight)),
  )
  return Hv
end

function Hvprod!(
  b::ForwardDiffADHvprod,
  Hv,
  x::AbstractVector{T},
  v,
  f,
  ::Val{:obj},
  obj_weight::Real = one(T),
) where {T}
  DI.hvp!(b.f, (Hv,), b.prep_obj, b.backend, x, (v,))
  Hv .*= obj_weight
  return Hv
end

# Hessian of the `j`-th nonlinear constraint: reuse the Lagrangian preparation with `y = eⱼ`
# and a zero objective weight.
function NLPModels.hprod!(
  b::ForwardDiffADHvprod,
  nlp::ADModel,
  x::AbstractVector{T},
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
) where {T}
  k = 0
  for i = 1:(nlp.meta.ncon)
    if i in nlp.meta.nln
      k += 1
      b.y[k] = i == j ? one(T) : zero(T)
    end
  end
  DI.hvp!(b.ℓ, (Hv,), b.prep_lag, b.backend, x, (v,), Cache(b.cx), Constant(b.y), Constant(zero(T)))
  return Hv
end

function NLPModels.hprod_residual!(
  b::ForwardDiffADHvprod,
  nls::AbstractADNLSModel,
  x::AbstractVector{T},
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
) where {T}
  for i = 1:(nls.nls_meta.nequ)
    b.y[i] = i == j ? one(T) : zero(T)
  end
  DI.hvp!(b.ℓ, (Hv,), b.prep_lag, b.backend, x, (v,), Cache(b.cx), Constant(b.y), Constant(zero(T)))
  return Hv
end

struct ForwardDiffADGHjvprod <: ADBackend end
function ForwardDiffADGHjvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ForwardDiffADGHjvprod()
end
function directional_second_derivative(::ForwardDiffADGHjvprod, f, x, v, w)
  return ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> f(x + s * w + t * v), 0), 0)
end
