export lls_autodiff

function lls_autodiff(::Type{T} = Float64) where {T}
  x0 = zeros(T, 2)
  F(x) = [x[1] - x[2]; x[1] + x[2] - 2; x[2] - 2]
  c(x) = [x[1] + x[2]]
  lcon = T[0.0]
  ucon = T[Inf]

  return ADNLSModel(F, x0, 3, c, lcon, ucon, name = "lls_autodiff")
end
