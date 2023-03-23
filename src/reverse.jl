struct ReverseDiffADGradient <: ADBackend
  cfg
end
struct ReverseDiffADJacobian <: ADBackend
  nnzj::Int
end
struct ReverseDiffADHessian <: ADBackend
  nnzh::Int
end
struct GenericReverseDiffADJprod <: ADBackend end
struct ReverseDiffADJtprod <: ADBackend end
struct ReverseDiffADHvprod <: ADBackend end

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

function ReverseDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ReverseDiffADJtprod()
end
function Jtprod!(::ReverseDiffADJtprod, Jtv, f, x, v)
  Jtv .= ReverseDiff.gradient(x -> dot(f(x), v), x)
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
function Hvprod!(::ReverseDiffADHvprod, Hv, f, x, v)
  Hv .= ForwardDiff.derivative(t -> ReverseDiff.gradient(f, x + t * v), 0)
  return Hv
end
