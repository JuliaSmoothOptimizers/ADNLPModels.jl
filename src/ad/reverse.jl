export ReverseDiffAD

struct ReverseDiffAD{T} <: ADBackend
  cfg::T
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