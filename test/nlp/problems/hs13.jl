export hs13_autodiff

hs13_autodiff(::Type{T}; kwargs...) where {T <: Number} = hs13_autodiff(Vector{T}; kwargs...)
function hs13_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  function f(x)
    return (x[1] - 2)^2 + x[2]^2
  end
  x0 = fill!(S(undef, 2), -2)
  lvar = fill!(S(undef, 2), 0)
  uvar = fill!(S(undef, 2), Inf)
  function c(x)
    return [(1 - x[1])^3 - x[2]]
  end
  lcon = fill!(S(undef, 1), 0)
  ucon = fill!(S(undef, 1), Inf)
  return ADNLPModels.ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon, name = "hs13_autodiff"; kwargs...)
end
