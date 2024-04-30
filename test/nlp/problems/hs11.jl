export hs11_autodiff

hs11_autodiff(::Type{T}; kwargs...) where {T <: Number} = hs11_autodiff(Vector{T}; kwargs...)
function hs11_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  x0 = S([49 // 10; 1 // 10])
  f(x) = (x[1] - 5)^2 + x[2]^2 - 25
  c(x) = [-x[1]^2 + x[2]]
  lcon = S([-Inf])
  ucon = S([0])

  return ADNLPModel(f, x0, c, lcon, ucon, name = "hs11_autodiff"; kwargs...)
end
