export ADBackend

abstract type ADBackend end

# include("forward.jl")
# include("reverse.jl")
export ForwardDiffAD
export ReverseDiffAD

struct ForwardDiffAD{T} <: ADBackend
  cfg::T
end

struct ReverseDiffAD{T} <: ADBackend
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

function ReverseDiffAD(f, x0::AbstractVector)
  f_tape = ReverseDiff.GradientTape(f, x0)
  cfg = ReverseDiff.compile(f_tape)
  return ReverseDiffAD{typeof(cfg)}(cfg)
end

function gradient!(ad::ReverseDiffAD, g, f, x)
  return ReverseDiff.gradient!(g, ad.cfg, x)
end

jacobian(::ReverseDiffAD, c, x) = ReverseDiff.jacobian(c, x)
hessian(::ReverseDiffAD, c, x) = ReverseDiff.hessian(c, x)

Jprod(::ReverseDiffAD, f, x, v) = vec(ReverseDiff.jacobian(t -> f(x + t[1] * v), [0.0]))
Jtprod(::ReverseDiffAD, f, x, v) = ReverseDiff.gradient(x -> dot(f(x), v), x)

function Hvprod(::ReverseDiffAD, f, x, v)
  return vec(ReverseDiff.jacobian(t -> ReverseDiff.gradient(f, x + t[1] * v), [0.0]))
end

function directional_second_derivative(::ReverseDiffAD, f, x, v, w)
  return w' * ReverseDiff.hessian(f, x) * v
end