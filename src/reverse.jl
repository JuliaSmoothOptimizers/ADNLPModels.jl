struct ReverseDiffADJacobian <: ADBackend
  nnzj::Int
end
struct ReverseDiffADHessian <: ADBackend
  nnzh::Int
end
struct GenericReverseDiffADJprod <: ADBackend end
struct GenericReverseDiffADJtprod <: ADBackend end

struct ReverseDiffADGradient{B, P} <: ADBackend
  backend::B
  prep::P
end

function ReverseDiffADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  @assert nvar > 0
  @lencheck nvar x0
  # `RDBackend` compiles the tape, matching the previous `ReverseDiff.compile` behaviour.
  prep = DI.prepare_gradient(f, RDBackend, x0; strict = Val(false))
  return ReverseDiffADGradient(RDBackend, prep)
end

function gradient!(b::ReverseDiffADGradient, g, f, x)
  return DI.gradient!(f, g, b.prep, b.backend, x)
end

struct GenericReverseDiffADGradient <: ADBackend end

function GenericReverseDiffADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  return GenericReverseDiffADGradient()
end

function gradient!(::GenericReverseDiffADGradient, g, f, x)
  return DI.gradient!(f, g, RDBackendNC, x)
end

function ReverseDiffADJacobian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzj = nvar * ncon
  return ReverseDiffADJacobian(nnzj)
end
jacobian(::ReverseDiffADJacobian, f, x) = DI.jacobian(f, RDBackendNC, x)

function ReverseDiffADHessian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzh = nvar * (nvar + 1) / 2
  return ReverseDiffADHessian(nnzh)
end
hessian(::ReverseDiffADHessian, f, x) = DI.hessian(f, RDBackendNC, x)

function GenericReverseDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericReverseDiffADJprod()
end
function Jprod!(::GenericReverseDiffADJprod, Jv, f, x, v, ::Val)
  DI.pushforward!(f, (Jv,), RDBackendNC, x, (v,))
  return Jv
end

struct ReverseDiffADJprod{B, P, S} <: InPlaceADbackend
  backend::B
  prep::P
  cx::S
end

function ReverseDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  cx = similar(x0, ncon)
  v0 = fill!(similar(x0, nvar), zero(T))
  prep = DI.prepare_pushforward(c!, cx, RDBackendNC, x0, (v0,); strict = Val(false))
  return ReverseDiffADJprod(RDBackendNC, prep, cx)
end

function Jprod!(b::ReverseDiffADJprod, Jv, c!, x, v, ::Val)
  DI.pushforward!(c!, b.cx, (Jv,), b.prep, b.backend, x, (v,))
  return Jv
end

function GenericReverseDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericReverseDiffADJtprod()
end
function Jtprod!(::GenericReverseDiffADJtprod, Jtv, f, x, v, ::Val)
  DI.gradient!(x -> dot(f(x), v), Jtv, RDBackendNC, x)
  return Jtv
end

struct ReverseDiffADJtprod{T, S, GT} <: InPlaceADbackend
  gtape::GT
  _tmp_out::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  _rval::S  # temporary storage for jtprod
end

function ReverseDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  _tmp_out = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, ncon)
  _rval = similar(x0, ncon)

  ψ(x, u; tmp_out = _tmp_out) = begin
    c!(tmp_out, x) # here x is a vector of ReverseDiff.TrackedReal
    dot(tmp_out, u)
  end
  u = fill!(similar(x0, ncon), zero(T)) # just for GradientConfig
  gcfg = ReverseDiff.GradientConfig((x0, u))
  gtape = ReverseDiff.compile(ReverseDiff.GradientTape(ψ, (x0, u), gcfg))

  return ReverseDiffADJtprod(gtape, _tmp_out, _rval)
end

function Jtprod!(b::ReverseDiffADJtprod, Jtv, c!, x, v, ::Val)
  ReverseDiff.gradient!((Jtv, b._rval), b.gtape, (x, v))
  return Jtv
end

struct GenericReverseDiffADHvprod <: ADBackend end

function GenericReverseDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericReverseDiffADHvprod()
end
function Hvprod!(::GenericReverseDiffADHvprod, Hv, x, v, f, args...)
  DI.hvp!(f, (Hv,), FoRBackendNC, x, (v,))
  return Hv
end

struct ReverseDiffADHvprod{T, S, Tagf, F, Tagψ, P} <: ADBackend
  z::Vector{ForwardDiff.Dual{Tagf, T, 1}}
  gz::Vector{ForwardDiff.Dual{Tagf, T, 1}}
  ∇f!::F
  zψ::Vector{ForwardDiff.Dual{Tagψ, T, 1}}
  yψ::Vector{ForwardDiff.Dual{Tagψ, T, 1}}
  gzψ::Vector{ForwardDiff.Dual{Tagψ, T, 1}}
  gyψ::Vector{ForwardDiff.Dual{Tagψ, T, 1}}
  ∇l!::P
  Hv_temp::S
end

function ReverseDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  # unconstrained Hessian
  tagf = ForwardDiff.Tag{typeof(f), T}
  z = seeded_duals(tagf, x0, nvar)
  gz = similar(z)
  f_tape = ReverseDiff.GradientTape(f, z)
  cfgf = ReverseDiff.compile(f_tape)
  ∇f!(gz, z; cfg = cfgf) = ReverseDiff.gradient!(gz, cfg, z)

  # constraints
  ψ(x, u) = begin # ; tmp_out = _tmp_out
    ncon = length(u)
    tmp_out = similar(x, ncon)
    c!(tmp_out, x)
    dot(tmp_out, u)
  end
  tagψ = ForwardDiff.Tag{typeof(ψ), T}
  zψ = seeded_duals(tagψ, x0, nvar)
  yψ = fill!(similar(zψ, ncon), zero(T))
  ψ_tape = ReverseDiff.GradientConfig((zψ, yψ))
  cfgψ = ReverseDiff.compile(ReverseDiff.GradientTape(ψ, (zψ, yψ), ψ_tape))

  gzψ = similar(zψ)
  gyψ = similar(yψ)
  function ∇l!(gz, gy, z, y; cfg = cfgψ)
    ReverseDiff.gradient!((gz, gy), cfg, (z, y))
  end
  Hv_temp = similar(x0)

  return ReverseDiffADHvprod(z, gz, ∇f!, zψ, yψ, gzψ, gyψ, ∇l!, Hv_temp)
end

function Hvprod!(
  b::ReverseDiffADHvprod{T, S, Tagf, F, Tagψ},
  Hv,
  x::AbstractVector{T},
  v,
  ℓ,
  ::Val{:lag},
  y,
  obj_weight::Real = one(T),
) where {T, S, Tagf, F, Tagψ}
  map!(ForwardDiff.Dual{Tagf}, b.z, x, v) # x + ε * v
  b.∇f!(b.gz, b.z) # ∇f(x + ε * v) = ∇f(x) + ε * ∇²f(x)ᵀv
  ForwardDiff.extract_derivative!(Tagf, Hv, b.gz)  # ∇²f(x)ᵀv
  Hv .*= obj_weight

  map!(ForwardDiff.Dual{Tagψ}, b.zψ, x, v)
  b.yψ .= y
  b.∇l!(b.gzψ, b.gyψ, b.zψ, b.yψ)
  ForwardDiff.extract_derivative!(Tagψ, b.Hv_temp, b.gzψ)
  Hv .+= b.Hv_temp

  return Hv
end

function Hvprod!(b::ReverseDiffADHvprod{T}, Hv, x::AbstractVector{T}, v, ci, ::Val{:ci}) where {T}
  Hv .= ForwardDiff.derivative(t -> ReverseDiff.gradient(ci, x + t * v), 0)
  return Hv
end

function Hvprod!(
  b::ReverseDiffADHvprod{T, S, Tagf},
  Hv,
  x,
  v,
  f,
  ::Val{:obj},
  obj_weight::Real = one(T),
) where {T, S, Tagf}
  map!(ForwardDiff.Dual{Tagf}, b.z, x, v) # x + ε * v
  b.∇f!(b.gz, b.z) # ∇f(x + ε * v) = ∇f(x) + ε * ∇²f(x)ᵀv
  ForwardDiff.extract_derivative!(Tagf, Hv, b.gz)  # ∇²f(x)ᵀv
  Hv .*= obj_weight
  return Hv
end
