function chainwoo_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n = 4 * max(1, div(n, 4))  # number of variables adjusted to be a multiple of 4
  function f(x)
    n = length(x)
    return 1 + sum(100 * (x[2*i]   - x[2*i-1]^2)^2 + (1 - x[2*i-1])^2 +
           90 * (x[2*i+2] - x[2*i+1]^2)^2 + (1 - x[2*i+1])^2 +
           10 * (x[2*i] + x[2*i+2] - 2)^2 + 0.1 * (x[2*i] - x[2*i+2])^2 for i=1:div(n,2)-1)
  end
  x0 = -2 * ones(T, n)
  return RADNLPModel(f, x0, name="chainwoo_radnlp"; kwargs...)
end

function chainwoo_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n = 4 * max(1, div(n, 4))  # number of variables adjusted to be a multiple of 4
  function f(x)
    n = length(x)
    return 1 + sum(100 * (x[2*i]   - x[2*i-1]^2)^2 + (1 - x[2*i-1])^2 +
           90 * (x[2*i+2] - x[2*i+1]^2)^2 + (1 - x[2*i+1])^2 +
           10 * (x[2*i] + x[2*i+2] - 2)^2 + 0.1 * (x[2*i] - x[2*i+2])^2 for i=1:div(n,2)-1)
  end
  x0 = -2 * ones(T, n)
  return ADNLPModel(f, x0, name="chainwoo_autodiff")
end
