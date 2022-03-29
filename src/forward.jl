struct ForwardDiffAD{T} <: ADBackend
  nnzh::Int
  nnzj::Int
  cfg::T
end

function ForwardDiffAD(
  nvar::Integer,
  f,
  ncon::Integer = 0;
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  @assert nvar > 0
  @lencheck nvar x0
  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon
  cfg = ForwardDiff.GradientConfig(f, x0)
  return ForwardDiffAD{typeof(cfg)}(nnzh, nnzj, cfg)
end

gradient(adbackend::ForwardDiffAD, f, x) = ForwardDiff.gradient(f, x, adbackend.cfg)
function gradient!(adbackend::ForwardDiffAD, g, f, x)
  return ForwardDiff.gradient!(g, f, x, adbackend.cfg)
end
jacobian(::ForwardDiffAD, f, x) = ForwardDiff.jacobian(f, x)
hessian(::ForwardDiffAD, f, x) = ForwardDiff.hessian(f, x)
function Jprod(::ForwardDiffAD, f, x, v)
  return ForwardDiff.derivative(t -> f(x + t * v), 0)
end
function Jtprod(::ForwardDiffAD, f, x, v)
  return ForwardDiff.gradient(x -> dot(f(x), v), x)
end
function Hvprod(::ForwardDiffAD, f, x, v)
  return ForwardDiff.derivative(t -> ForwardDiff.gradient(f, x + t * v), 0)
end
