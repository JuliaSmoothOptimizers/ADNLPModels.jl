export nlshs20_autodiff

nlshs20_autodiff(::Type{T}; kwargs...) where {T <: Number} = nlshs20_autodiff(Vector{T}; kwargs...)
function nlshs20_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  x0 = S([-2; 1])
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]
  lvar = S([-1 // 2; -Inf])
  uvar = S([1 // 2; Inf])
  c(x) = [x[1] + x[2]^2; x[1]^2 + x[2]; x[1]^2 + x[2]^2 - 1]
  lcon = fill!(S(undef, 3), 0)
  ucon = fill!(S(undef, 3), Inf)

  return ADNLSModel(F, x0, 2, lvar, uvar, c, lcon, ucon, name = "nlshs20_autodiff"; kwargs...)
end
