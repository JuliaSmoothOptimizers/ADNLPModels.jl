function broydn7d_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n2 = max(1, div(n, 2))
  n = 2 * n2  # number of variables adjusted to be even
  function f(x)
    n = length(x)
    p = 7/3
    return abs(1 - 2 * x[2] + (3 - x[1] / 2) * x[1])^p +
           sum(abs(1 - x[i-1] - 2 * x[i+1] + (3 - x[i] / 2) * x[i])^p for i=2:n-1) +
           abs(1 - x[n-1] + (3 - x[n] / 2) * x[n])^p +
           sum(abs(x[i] + x[i + n2])^p for i=1:n2)
  end
  x0 = -ones(T, n)
  return RADNLPModel(f, x0, name="broydn7d_radnlp"; kwargs...)
end

function broydn7d_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n2 = max(1, div(n, 2))
  n = 2 * n2  # number of variables adjusted to be even
  function f(x)
    n = length(x)
    p = 7/3
    return abs(1 - 2 * x[2] + (3 - x[1] / 2) * x[1])^p +
           sum(abs(1 - x[i-1] - 2 * x[i+1] + (3 - x[i] / 2) * x[i])^p for i=2:n-1) +
           abs(1 - x[n-1] + (3 - x[n] / 2) * x[n])^p +
           sum(abs(x[i] + x[i + n2])^p for i=1:n2)
  end
  x0 = -ones(T, n)
  return ADNLPModel(f, x0, name="broydn7d_autodiff")
end
