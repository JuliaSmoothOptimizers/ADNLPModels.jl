function penalty2_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 3 || error("penalty2 : n ≥ 3")
  function f(x)
    n = length(x)
    a = 1.0e-5
    m = 2 * n
    y = ones(m)
    for i = 1:m
      y[i] = exp(i / 10.0) + exp((i-1) / 10.0)
    end
    return (x[1] - 0.2)^2 +
      sum(a * (exp(x[i] / 10.0) + exp(x[i-1] / 10.0) - y[i])^2 for i=2:n) +
      sum(a * (exp(x[i-n+1] / 10.0) - exp(-1/10))^2 for i=n+1:2*n-1) +
      (sum((n-j+1) * x[j]^2  for j=1:n) - 1.0)^2
  end
  x0 = ones(T, n) / 2
  return RADNLPModel(f, x0, name="penalty2_radnlp"; kwargs...)
end

function penalty2_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 3 || error("penalty2 : n ≥ 3")
  function f(x)
    n = length(x)
    a = 1.0e-5
    m = 2 * n
    y = ones(m)
    for i = 1:m
      y[i] = exp(i / 10.0) + exp((i-1) / 10.0)
    end
    return (x[1] - 0.2)^2 +
      sum(a * (exp(x[i] / 10.0) + exp(x[i-1] / 10.0) - y[i])^2 for i=2:n) +
      sum(a * (exp(x[i-n+1] / 10.0) - exp(-1/10))^2 for i=n+1:2*n-1) +
      (sum((n-j+1) * x[j]^2  for j=1:n) - 1.0)^2
  end
  x0 = ones(T, n) / 2
  return ADNLPModel(f, x0, name="penalty2_autodiff")
end
