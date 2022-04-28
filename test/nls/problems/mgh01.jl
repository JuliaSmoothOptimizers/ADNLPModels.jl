export mgh01_autodiff # , MGH01_special

function mgh01_autodiff(::Type{T} = Float64; kwargs...) where {T}
  x0 = T[-1.2; 1.0]
  F(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]

  return ADNLSModel(F, x0, 2, name = "mgh01_autodiff"; kwargs...)
end

# MGH01_special() = FeasibilityResidual(MGH01Feas())
