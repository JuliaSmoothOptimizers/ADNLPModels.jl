function power_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return (sum((i * x[i]^2) for i=1:n))^2
  end
  x0 = ones(T, n)
  return RADNLPModel(f, x0, name="power_radnlp"; kwargs...)
end

function power_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return (sum((i * x[i]^2) for i=1:n))^2
  end
  x0 = ones(T, n)
  return ADNLPModel(f, x0, name="power_autodiff")
end
