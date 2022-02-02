export ForwardDiffAD

struct ForwardDiffAD{T} <: ADBackend
  cfg::T
end

function ForwardDiffAD(f, x0::AbstractVector)
  cfg = ForwardDiff.GradientConfig(f, x0)
  return ForwardDiffAD{typeof(cfg)}(cfg)
end

function gradient!(ad::ForwardDiffAD, g, f, x)
  return ForwardDiff.gradient!(g, f, x, ad.cfg)
end

jacobian(::ForwardDiffAD, f, x) = ForwardDiff.jacobian(f, x)
hessian(::ForwardDiffAD, f, x) = ForwardDiff.hessian(f, x)

Jprod(::ForwardDiffAD, f, x, v) = ForwardDiff.derivative(t -> f(x + t * v), 0)
Jtprod(::ForwardDiffAD, f, x, v) = ForwardDiff.gradient(x -> dot(f(x), v), x)

function Hvprod(::ForwardDiffAD, f, x, v)
  return ForwardDiff.derivative(t -> ForwardDiff.gradient(f, x + t * v), 0)
end

function directional_second_derivative(::ForwardDiffAD, f, x, v, w)
  return ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> f(x + s * w + t * v), 0), 0)
end