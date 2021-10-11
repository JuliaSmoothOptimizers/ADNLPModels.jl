export hs14_autodiff

function hs14_autodiff(::Type{T} = Float64) where {T}
  x0 = T[2.0; 2.0]
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  c(x) = [x[1] - 2 * x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1]
  lcon = T[0.0; 0.0]
  ucon = T[0.0; Inf]

  return ADNLPModel(f, x0, c, lcon, ucon, name = "hs14_autodiff")
end
