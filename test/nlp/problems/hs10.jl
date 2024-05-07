export hs10_autodiff

hs10_autodiff(::Type{T}; kwargs...) where {T <: Number} = hs10_autodiff(Vector{T}; kwargs...)
function hs10_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  x0 = S([-10; 10])
  f(x) = x[1] - x[2]
  c(x) = [-3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 + 1]
  lcon = S([0])
  ucon = S([Inf])

  return ADNLPModel(f, x0, c, lcon, ucon, name = "hs10_autodiff"; kwargs...)
end
