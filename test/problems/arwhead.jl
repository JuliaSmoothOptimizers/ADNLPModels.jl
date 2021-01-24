function arwhead_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return sum((x[i]^2 + x[n]^2)^2 - 4 * x[i] + 3 for i=1:n-1)
  end
  x0 = ones(T, n)
  return RADNLPModel(f, x0, name="arwhead_radnlp"; kwargs...)
end

function arwhead_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return sum((x[i]^2 + x[n]^2)^2 - 4 * x[i] + 3 for i=1:n-1)
  end
  x0 = ones(T, n)
  return ADNLPModel(f, x0, name="arwhead_autodiff")
end
