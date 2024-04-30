export mgh01_autodiff # , MGH01_special

mgh01_autodiff(::Type{T}; kwargs...) where {T <: Number} = mgh01_autodiff(Vector{T}; kwargs...)
function mgh01_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  x0 = S([-12 // 10; 1])
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]

  return ADNLSModel(F, x0, 2, name = "mgh01_autodiff"; kwargs...)
end

# MGH01_special() = FeasibilityResidual(MGH01Feas())
