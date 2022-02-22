export hs13_autodiff

function hs13_autodiff(::Type{T} = Float64; kwargs...) where {T}
  function f(x)
    return (x[1] - 2)^2 + x[2]^2
  end
  x0 = -2 * ones(T, 2)
  lvar = zeros(T, 2)
  uvar = T(Inf) * ones(T, 2)
  function c(x)
    return [(1 - x[1])^3 - x[2]]
  end
  lcon = zeros(T, 1)
  ucon = T(Inf) * ones(T, 1)
  return ADNLPModels.ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon, name = "hs13_autodiff"; kwargs...)
end
