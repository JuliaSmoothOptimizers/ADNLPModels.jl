export lls_autodiff

lls_autodiff(::Type{T}; kwargs...) where {T <: Number} = lls_autodiff(Vector{T}; kwargs...)
function lls_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  x0 = fill!(S(undef, 2), 0)
  F(x) = [x[1] - x[2]; x[1] + x[2] - 2; x[2] - 2]
  lcon = S([0])
  ucon = S([Inf])

  clinrows = [1, 1]
  clincols = [1, 2]
  clinvals = S([1, 1])

  return ADNLSModel(
    F,
    x0,
    3,
    clinrows,
    clincols,
    clinvals,
    lcon,
    ucon,
    name = "lls_autodiff";
    kwargs...,
  )
end
