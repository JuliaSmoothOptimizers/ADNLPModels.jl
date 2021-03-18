function tridia_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T

  function f(x)
    n = length(x)
    return (x[1] - 1)^2 + sum(i * (- x[i-1] + 2 * x[i])^2 for i=2:n)
  end
  x0 = ones(T, n)
  return RADNLPModel(f, x0, name="tridia_radnlp"; kwargs...)
end

function tridia_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T

  function f(x)
    n = length(x)
    return (x[1] - 1)^2 + sum(i * (- x[i-1] + 2 * x[i])^2 for i=2:n)
  end
  x0 = ones(T, n)
  return ADNLPModel(f, x0, name="tridia_autodiff")
end
