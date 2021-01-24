function cragglvy_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 2 || error("cragglvy : n ≥ 2")
  function f(x)
    n = length(x)
    return sum((exp(x[2*i-1]) - x[2*i])^4 + 100 * (x[2*i] - x[2*i+1])^6 +
           (tan(x[2*i+1] - x[2*i+2]) + x[2*i+1] - x[2*i+2])^4 +
           x[2*i-1]^8 + (x[2*i+2] - 1)^2 for i = 1:div(n,2)-1)
  end
  x0 = 2 * ones(T, n)
  return RADNLPModel(f, x0, name="cragglvy_radnlp"; kwargs...)
end

function cragglvy_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 2 || error("cragglvy : n ≥ 2")
  function f(x)
    n = length(x)
    return sum((exp(x[2*i-1]) - x[2*i])^4 + 100 * (x[2*i] - x[2*i+1])^6 +
           (tan(x[2*i+1] - x[2*i+2]) + x[2*i+1] - x[2*i+2])^4 +
           x[2*i-1]^8 + (x[2*i+2] - 1)^2 for i = 1:div(n,2)-1)
  end
  x0 = 2 * ones(T, n)
  return ADNLPModel(f, x0, name="cragglvy_autodiff")
end
