function morebv_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 2 || error("morebv : n ≥ 2")
  function f(x)
    n = length(x)
    h = 1.0/(n-1)
    return sum((2.0 * x[i] - x[i-1] - x[i+1] + (h^2 / 2.0) * (x[i] + (i - 1) * h + 1)^3)^2 for i=2:n-1)
  end

  x0 = ones(T, n) / 2
  x0[1] = zero(T)
  x0[n] = zero(T)

  return RADNLPModel(f, x0, name="morebv_radnlp"; kwargs...)
end

function morebv_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 2 || error("morebv : n ≥ 2")
  function f(x)
    n = length(x)
    h = 1.0/(n-1)
    return sum((2.0 * x[i] - x[i-1] - x[i+1] + (h^2 / 2.0) * (x[i] + (i - 1) * h + 1)^3)^2 for i=2:n-1)
  end

  x0 = ones(T, n) / 2
  x0[1] = zero(T)
  x0[n] = zero(T)

  return ADNLPModel(f, x0, name="morebv_autodiff")
end
