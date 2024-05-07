export hs14_autodiff

hs14_autodiff(::Type{T}; kwargs...) where {T <: Number} = hs14_autodiff(Vector{T}; kwargs...)
function hs14_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  x0 = S([2; 2])
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  c(x) = [-x[1]^2 / 4 - x[2]^2 + 1]
  lcon = S([-1; 0])
  ucon = S([-1; Inf])

  clinrows = [1, 1]
  clincols = [1, 2]
  clinvals = S([1, -2])

  return ADNLPModel(
    f,
    x0,
    clinrows,
    clincols,
    clinvals,
    c,
    lcon,
    ucon,
    name = "hs14_autodiff";
    kwargs...,
  )
end
