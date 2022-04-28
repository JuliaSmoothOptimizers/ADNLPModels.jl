export linsv_autodiff

function linsv_autodiff(::Type{T} = Float64; kwargs...) where {T}
  x0 = zeros(T, 2)
  f(x) = x[1]
  lcon = T[3.0; 1.0]
  ucon = T[Inf; Inf]

  clinrows = [1, 1, 2]
  clincols = [1, 2, 2]
  clinvals = T[1, 1, 1]

  return ADNLPModel(f, x0, clinrows, clincols, clinvals, lcon, ucon, name = "linsv_autodiff", lin = collect(1:2); kwargs...)
end
