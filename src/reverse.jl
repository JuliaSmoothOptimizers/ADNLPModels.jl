struct ReverseDiffADGradient <: ADBackend
  cfg
end
struct ReverseDiffADJacobian <: ADBackend
  nnzj::Int
end
struct ReverseDiffADHessian <: ADBackend
  nnzh::Int
end
struct ReverseDiffADJprod <: ADBackend end
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

function ReverseDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ReverseDiffADJprod()
end
function Jprod!(::ReverseDiffADJprod, Jv, f, x, v)
  Jv .= vec(ReverseDiff.jacobian(t -> f(x + t[1] * v), [0.0]))
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
