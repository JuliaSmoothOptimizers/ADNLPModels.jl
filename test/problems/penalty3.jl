function penalty3_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 3 || error("penalty3 : n ≥ 3")
  function f(x)
    n = length(x)
    return 1.0 + sum((x[i] - 1.0)^2 for i=1:div(n,2)) +
      exp(x[n]) * sum((x[i] + 2.0 * x[i+1] + 10.0 * x[i+2] - 1.0)^2 for i=1:n-2) +
      sum((x[i] + 2.0 * x[i+1] + 10.0 * x[i+2] - 1.0)^2 for i=1:n-2) * sum((2.0 * x[i] + x[i+1] - 3.0)^2 for i=1:n-2) +
      exp(x[n-1]) * sum((2.0 * x[i] + x[i+1] - 3.0)^2 for i=1:n-2) +
      (sum(x[i]^2 - n for i=1:n))^2
  end
  x0 = T.([i/(n+1) for i=1:n])
  return RADNLPModel(f, x0, name="penalty3_radnlp"; kwargs...)
end

function penalty3_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 3 || error("penalty3 : n ≥ 3")
  function f(x)
    n = length(x)
    return 1.0 + sum((x[i] - 1.0)^2 for i=1:div(n,2)) +
      exp(x[n]) * sum((x[i] + 2.0 * x[i+1] + 10.0 * x[i+2] - 1.0)^2 for i=1:n-2) +
      sum((x[i] + 2.0 * x[i+1] + 10.0 * x[i+2] - 1.0)^2 for i=1:n-2) * sum((2.0 * x[i] + x[i+1] - 3.0)^2 for i=1:n-2) +
      exp(x[n-1]) * sum((2.0 * x[i] + x[i+1] - 3.0)^2 for i=1:n-2) +
      (sum(x[i]^2 - n for i=1:n))^2
  end
  x0 = T.([i/(n+1) for i=1:n])
  return ADNLPModel(f, x0, name="penalty3_autodiff")
end
