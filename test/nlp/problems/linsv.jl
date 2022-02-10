export linsv_autodiff

function linsv_autodiff(::Type{T} = Float64; kwargs...) where {T}
  x0 = zeros(T, 2)
  f(x) = x[1]
  con(x) = [x[1] + x[2]; x[2]]
  lcon = T[3.0; 1.0]
  ucon = T[Inf; Inf]

  return ADNLPModel(f, x0, con, lcon, ucon, name = "linsv_autodiff"; kwargs...)
end
