export nlshs20_autodiff

function nlshs20_autodiff(::Type{T} = Float64; kwargs...) where {T}
  x0 = T[-2.0; 1.0]
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]
  lvar = T[-0.5; -Inf]
  uvar = T[0.5; Inf]
  c(x) = [x[1] + x[2]^2; x[1]^2 + x[2]; x[1]^2 + x[2]^2 - 1]
  lcon = zeros(T, 3)
  ucon = fill(T(Inf), 3)

  return ADNLSModel(F, x0, 2, lvar, uvar, c, lcon, ucon, name = "nlshs20_autodiff"; kwargs...)
end
