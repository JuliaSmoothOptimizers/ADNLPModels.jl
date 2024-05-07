export mgh01feas_autodiff

mgh01feas_autodiff(::Type{T}; kwargs...) where {T <: Number} = mgh01feas_autodiff(Vector{T}; kwargs...)
function mgh01feas_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  x0 = S([-12 // 10; 1])
  f(x) = zero(eltype(x))
  c(x) = [10 * (x[2] - x[1]^2)]
  lcon = S([1, 0])
  ucon = S([1, 0])

  clinrows = [1]
  clincols = [1]
  clinvals = S([1])

  return ADNLPModel(
    f,
    x0,
    clinrows,
    clincols,
    clinvals,
    c,
    lcon,
    ucon,
    name = "mgh01feas_autodiff";
    kwargs...,
  )
end
