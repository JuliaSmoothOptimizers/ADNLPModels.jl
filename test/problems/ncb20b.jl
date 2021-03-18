function ncb20b_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 20 || error("ncb20 : n ≥ 20")
  function f(x)
    n = length(x)
    h = 1.0/(n-1)
    return sum((10.0 / i) * (sum(x[i+j-1] / (1 + x[i+j-1]^2) for j=1:20))^2 - 0.2 * sum(x[i+j-1] for j=1:20) for i=1:n-19) +
      sum(100.0 * x[i]^4 + 2.0 for i=1:n)
  end
  x0 = zeros(T, n)
  return RADNLPModel(f, x0, name="ncb20b_radnlp"; kwargs...)
end

function ncb20b_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 20 || error("ncb20 : n ≥ 20")
  function f(x)
    n = length(x)
    h = 1.0/(n-1)
    return sum((10.0 / i) * (sum(x[i+j-1] / (1 + x[i+j-1]^2) for j=1:20))^2 - 0.2 * sum(x[i+j-1] for j=1:20) for i=1:n-19) +
      sum(100.0 * x[i]^4 + 2.0 for i=1:n)
  end
  x0 = zeros(T, n)
  return ADNLPModel(f, x0, name="ncb20b_autodiff")
end
