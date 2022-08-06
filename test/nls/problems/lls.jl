export lls_autodiff

function lls_autodiff(::Type{T} = Float64; kwargs...) where {T}
  x0 = zeros(T, 2)
  F(x) = [x[1] - x[2]; x[1] + x[2] - 2; x[2] - 2]
  lcon = T[0.0]
  ucon = T[Inf]

  clinrows = [1, 1]
  clincols = [1, 2]
  clinvals = T[1, 1]

  return ADNLSModel(
    F,
    x0,
    3,
    clinrows,
    clincols,
    clinvals,
    lcon,
    ucon,
    name = "lls_autodiff";
    kwargs...,
  )
end
