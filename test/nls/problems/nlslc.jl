export nlslc_autodiff

nlslc_autodiff(::Type{T}; kwargs...) where {T <: Number} = nlslc_autodiff(Vector{T}; kwargs...)
function nlslc_autodiff(::Type{S} = Vector{Float64}; kwargs...) where {S}
  T = eltype(S)
  A = T[1 2; 3 4]
  b = T[5; 6]
  B = diagm(T[3 * i for i = 3:5])
  c = T[1; 2; 3]
  C = T[0 -2; 4 0]
  d = T[1; -1]

  x0 = fill!(S(undef, 15), 0)
  F(x) = [x[i]^2 - i^2 for i = 1:15]

  lcon = S([22.0; 1.0; -Inf; -11.0; -d; -b; -Inf * ones(3)])
  ucon = S([22.0; Inf; 16.0; 9.0; -d; Inf * ones(2); c])

  clinrows = [1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 7, 8, 9, 10, 11]
  clincols = [15, 10, 11, 12, 13, 14, 8, 9, 7, 6, 1, 1, 2, 2, 3, 4, 5]
  clinvals = S(vcat(T(15), c, d, b, C[1, 2], C[2, 1], A[:], diag(B)))

  return ADNLSModel(
    F,
    x0,
    15,
    clinrows,
    clincols,
    clinvals,
    lcon,
    ucon,
    name = "nlslincon_autodiff";
    kwargs...,
  )
end
