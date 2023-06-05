struct ReverseDiffADJacobian <: ADBackend
  nnzj::Int
end
struct ReverseDiffADHessian <: ADBackend
  nnzh::Int
end
struct GenericReverseDiffADJprod <: ADBackend end
struct GenericReverseDiffADJtprod <: ADBackend end
struct ReverseDiffADHvprod <: ADBackend end

struct ReverseDiffADGradient <: ADBackend
  cfg
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
  f_tape = ReverseDiff.GradientTape(f, x0)
  cfg = ReverseDiff.compile(f_tape)
  return ReverseDiffADGradient(cfg)
end

function gradient!(adbackend::ReverseDiffADGradient, g, f, x)
  return ReverseDiff.gradient!(g, adbackend.cfg, x)
end

struct GenericReverseDiffADGradient <: ADNLPModels.ADBackend end

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

function ADNLPModels.gradient!(::GenericReverseDiffADGradient, g, f, x)
  return ReverseDiff.gradient!(g, f, x)
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
jacobian(::ReverseDiffADJacobian, f, x) = ReverseDiff.jacobian(f, x)

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
hessian(::ReverseDiffADHessian, f, x) = ReverseDiff.hessian(f, x)

function GenericReverseDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericReverseDiffADJprod()
end
function Jprod!(::GenericReverseDiffADJprod, Jv, f, x, v)
  Jv .= vec(ReverseDiff.jacobian(t -> f(x + t[1] * v), [0.0]))
  return Jv
end

struct ReverseDiffADJprod{T, S, F} <: InPlaceADbackend
  ϕ!::F
  tmp_in::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  tmp_out::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  _tmp_out::S
  z::Vector{T}
end

function ReverseDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  tmp_in = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, nvar)
  tmp_out = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, ncon)
  _tmp_out = similar(x0, ncon)
  z = [zero(T)]

  # ... auxiliary function for J(x) * v
  # ... J(x) * v is the derivative at t = 0 of t ↦ r(x + tv)
  ϕ!(out, t; x = x0, v = x0, tmp_in = tmp_in, c! = c!) = begin
    # here t is a vector of ReverseDiff.TrackedReal
    tmp_in .= (t[1] .* v .+ x)
    c!(out, tmp_in)
    out
  end

  return ReverseDiffADJprod(ϕ!, tmp_in, tmp_out, _tmp_out, z)
end

function Jprod!(b::ReverseDiffADJprod, Jv, c!, x, v)
  ReverseDiff.jacobian!(Jv, (out, t) -> b.ϕ!(out, t, x = x, v = v), b._tmp_out, b.z)
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
function Jtprod!(::GenericReverseDiffADJtprod, Jtv, f, x, v)
  Jtv .= ReverseDiff.gradient(x -> dot(f(x), v), x)
  return Jtv
end

struct ReverseDiffADJtprod{T, S, GT} <: ADNLPModels.InPlaceADbackend
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

function ADNLPModels.Jtprod!(b::ReverseDiffADJtprod, Jtv, c!, x, v)
  ReverseDiff.gradient!((Jtv, b._rval), b.gtape, (x, v))
  return Jtv
end

function ReverseDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ReverseDiffADHvprod()
end
function Hvprod!(::ReverseDiffADHvprod, Hv, x, v, f, args...)
  Hv .= ForwardDiff.derivative(t -> ReverseDiff.gradient(f, x + t * v), 0)
  return Hv
end
