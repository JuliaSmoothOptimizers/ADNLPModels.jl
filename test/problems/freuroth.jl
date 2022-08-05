function freuroth_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return sum(((5 - x[i+1]) * x[i+1]^2 + x[i] - 2 * x[i+1] - 13)^2 for i=1:n-1) + sum(((1 + x[i+1]) * x[i+1]^2 + x[i] - 14 * x[i+1] - 29)^2 for i=1:n-1)
  end

  x0 = zeros(T, n)
  x0[1] = one(T) / 2
  x0[2] = -2 * one(T)
  return RADNLPModel(f, x0, name="freuroth_radnlp"; kwargs...)
end

function freuroth_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return sum(((5 - x[i+1]) * x[i+1]^2 + x[i] - 2 * x[i+1] - 13)^2 for i=1:n-1) + sum(((1 + x[i+1]) * x[i+1]^2 + x[i] - 14 * x[i+1] - 29)^2 for i=1:n-1)
  end

  x0 = zeros(T, n)
  x0[1] = one(T) / 2
  x0[2] = -2 * one(T)
  return ADNLPModel(f, x0, name="freuroth_autodiff")
end
