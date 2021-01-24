function brybnd_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    ml = 5
    mu = 1
    return sum((x[i] * (2 + 5 * x[i]^2) + 1 - sum(x[j] * (1 + x[j]) for j = max(1, i-ml) : min(n, i+mu) if j != i))^2 for i=1:n)
  end
  x0 = -ones(T, n)
  return RADNLPModel(f, x0, name="brybnd_radnlp"; kwargs...)
end

function brybnd_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    ml = 5
    mu = 1
    return sum((x[i] * (2 + 5 * x[i]^2) + 1 - sum(x[j] * (1 + x[j]) for j = max(1, i-ml) : min(n, i+mu) if j != i))^2 for i=1:n)
  end
  x0 = -ones(T, n)
  return ADNLPModel(f, x0, name="brybnd_autodiff")
end
