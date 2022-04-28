export hs14_autodiff

function hs14_autodiff(::Type{T} = Float64; kwargs...) where {T}
  x0 = T[2.0; 2.0]
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  c(x) = [-x[1]^2 / 4 - x[2]^2 + 1]
  lcon = T[-1; 0.0]
  ucon = T[-1; Inf]

  clinrows = [1, 1]
  clincols = [1, 2]
  clinvals = T[1, -2]

  return ADNLPModel(f, x0, clinrows, clincols, clinvals, c, lcon, ucon, name = "hs14_autodiff"; kwargs...)
end
