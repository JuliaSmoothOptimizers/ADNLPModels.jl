export bndrosenbrock_autodiff

function bndrosenbrock_autodiff(::Type{T} = Float64; kwargs...) where {T}
  x0 = T[-1.2; 1]
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]

  lvar = T[-1; -2]
  uvar = T[0.8; 2]

  return ADNLSModel(
    F,
    x0,
    2,
    lvar,
    uvar,
    name = "bndrosenbrock_autodiff";
    kwargs...,
  )
end
