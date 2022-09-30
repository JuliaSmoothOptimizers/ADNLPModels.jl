struct ForwardDiffADGradient <: ADBackend
  cfg
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
  cfg = ForwardDiff.GradientConfig(f, x0)
  return ForwardDiffADGradient(cfg)
end
gradient(adbackend::ForwardDiffADGradient, f, x) = ForwardDiff.gradient(f, x, adbackend.cfg)
function gradient!(adbackend::ForwardDiffADGradient, g, f, x)
  return ForwardDiff.gradient!(g, f, x, adbackend.cfg)
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
jacobian(::ForwardDiffADJacobian, f, x) = ForwardDiff.jacobian(f, x)

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
hessian(::ForwardDiffADHessian, f, x) = ForwardDiff.hessian(f, x)

struct ForwardDiffADJprod <: ADBackend end
function ForwardDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ForwardDiffADJprod()
end
function Jprod(::ForwardDiffADJprod, f, x, v)
  return ForwardDiff.derivative(t -> f(x + t * v), 0)
end

struct ForwardDiffADJtprod <: ADBackend end
function ForwardDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ForwardDiffADJtprod()
end
function Jtprod(::ForwardDiffADJtprod, f, x, v)
  return ForwardDiff.gradient(x -> dot(f(x), v), x)
end

struct ForwardDiffADHvprod{S} <: ADBackend
  gx::S
end

function ForwardDiffADHvprod(nvar::Integer, f, ncon::Integer = 0, c::Function = (args...) -> []; x0::AbstractVector = rand(nvar), kwargs...)
  T = eltype(x0)
  gx = zeros(T, nvar)
  return ForwardDiffADHvprod(gx)
end

function Hvprod!(Hv, b::ForwardDiffADHvprod{S}, f, x, v) where {S}
  ForwardDiff.derivative!(Hv, (y, t) -> ForwardDiff.gradient!(y, f, x + t * v), b.gx, 0)
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
