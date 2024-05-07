export brownden_autodiff

brownden_autodiff(::Type{T}; kwargs...) where {T <: Number} =
  brownden_autodiff(Vector{T}; kwargs...)
function brownden_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  T = eltype(S)
  x0 = S([25.0; 5.0; -5.0; -1.0])
  f(x) = begin
    s = zero(T)
    for i = 1:20
      s +=
        (
          (x[1] + x[2] * T(i) / 5 - exp(T(i) / 5))^2 +
          (x[3] + x[4] * sin(T(i) / 5) - cos(T(i) / 5))^2
        )^2
    end
    return s
  end

  return ADNLPModel(f, x0, name = "brownden_autodiff"; kwargs...)
end
