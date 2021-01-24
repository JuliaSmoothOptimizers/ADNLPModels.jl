function noncvxun_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 2 || error("noncvxun : n ≥ 2")
  function f(x)
    n = length(x)
    return sum((x[i] + x[mod(2*i-1, n) + 1] + x[mod(3*i-1, n) + 1])^2 +
      4.0 * cos(x[i] + x[mod(2*i-1, n) + 1] + x[mod(3*i-1, n) + 1]) for i=1:n)
  end
  x0 = T.([i for i=1:n])
  return RADNLPModel(f, x0, name="noncvxun_radnlp"; kwargs...)
end

function noncvxun_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 2 || error("noncvxun : n ≥ 2")
  function f(x)
    n = length(x)
    return sum((x[i] + x[mod(2*i-1, n) + 1] + x[mod(3*i-1, n) + 1])^2 +
      4.0 * cos(x[i] + x[mod(2*i-1, n) + 1] + x[mod(3*i-1, n) + 1]) for i=1:n)
  end
  x0 = T.([i for i=1:n])
  return ADNLPModel(f, x0, name="noncvxun_autodiff")
end
