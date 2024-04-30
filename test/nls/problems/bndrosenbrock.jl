export bndrosenbrock_autodiff

bndrosenbrock_autodiff(::Type{T}; kwargs...) where {T <: Number} = bndrosenbrock_autodiff(Vector{T}; kwargs...)
function bndrosenbrock_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  x0 = S([-12 // 10; 1])
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]

  lvar = S([-1; -2])
  uvar = S([8 // 10; 2])

  return ADNLSModel(F, x0, 2, lvar, uvar, name = "bndrosenbrock_autodiff"; kwargs...)
end
