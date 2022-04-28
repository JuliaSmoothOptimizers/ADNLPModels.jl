export mgh01feas_autodiff

function mgh01feas_autodiff(::Type{T} = Float64; kwargs...) where {T}
  x0 = T[-1.2; 1.0]
  f(x) = zero(eltype(x))
  c(x) = [10 * (x[2] - x[1]^2)]
  lcon = T[1, 0]
  ucon = T[1, 0]

  clinrows = [1]
  clincols = [1]
  clinvals = T[1]

  return ADNLPModel(f, x0, clinrows, clincols, clinvals, c, lcon, ucon, name = "mgh01feas_autodiff"; kwargs...)
end
