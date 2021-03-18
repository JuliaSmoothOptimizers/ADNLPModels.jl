function scosine_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 2 || error("scosine : n ≥ 2")
  p = zeros(n)
  for i=1:n
    p[i] = exp(6.0 * (i-1) / (n-1))
  end
  function f(x)
    n = length(x)
    return sum(cos(p[i]^2 * x[i]^2 - p[i+1] * x[i+1] / 2.0) for i=1:n-1)
  end
  x0 = T.([1/p[i] for i=1:n])
  return RADNLPModel(f, x0, name="scosine_radnlp"; kwargs...)
end

function scosine_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 2 || error("scosine : n ≥ 2")
  p = zeros(n)
  for i=1:n
    p[i] = exp(6.0 * (i-1) / (n-1))
  end
  function f(x)
    n = length(x)
    return sum(cos(p[i]^2 * x[i]^2 - p[i+1] * x[i+1] / 2.0) for i=1:n-1)
  end
  x0 = T.([1/p[i] for i=1:n])
  return ADNLPModel(f, x0, name="scosine_autodiff")
end
