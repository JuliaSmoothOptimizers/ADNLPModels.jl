export hs6_autodiff

hs6_autodiff(::Type{T}; kwargs...) where {T <: Number} = hs6_autodiff(Vector{T}; kwargs...)
function hs6_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  x0 = S([-12 // 10; 1])
  f(x) = (1 - x[1])^2
  c(x) = [10 * (x[2] - x[1]^2)]
  lcon = fill!(S(undef, 1), 0)
  ucon = fill!(S(undef, 1), 0)

  return ADNLPModel(f, x0, c, lcon, ucon, name = "hs6_autodiff"; kwargs...)
end
