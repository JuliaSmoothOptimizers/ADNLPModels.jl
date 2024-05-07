export linsv_autodiff

linsv_autodiff(::Type{T}; kwargs...) where {T <: Number} = linsv_autodiff(Vector{T}; kwargs...)
function linsv_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  x0 = fill!(S(undef, 2), 0)
  f(x) = x[1]
  lcon = S([3; 1])
  ucon = S([Inf; Inf])

  clinrows = [1, 1, 2]
  clincols = [1, 2, 2]
  clinvals = S([1, 1, 1])

  return ADNLPModel(
    f,
    x0,
    clinrows,
    clincols,
    clinvals,
    lcon,
    ucon,
    name = "linsv_autodiff",
    lin = collect(1:2);
    kwargs...,
  )
end
