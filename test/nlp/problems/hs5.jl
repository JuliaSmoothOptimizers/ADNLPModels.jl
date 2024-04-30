export hs5_autodiff

hs5_autodiff(::Type{T}; kwargs...) where {T <: Number} = hs5_autodiff(Vector{T}; kwargs...)
function hs5_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  x0 = fill!(S(undef, 2), 0)
  f(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 3x[1] / 2 + 5x[2] / 2 + 1
  l = S([-1.5; -3.0])
  u = S([4.0; 3.0])

  return ADNLPModel(f, x0, l, u, name = "hs5_autodiff"; kwargs...)
end
