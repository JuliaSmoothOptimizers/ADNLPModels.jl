function genrose_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return 1 + 100 * sum((x[i+1] - x[i]^2)^2 for i=1:n-1) + sum((x[i] - 1)^2 for  i=1:n-1)
  end
  x0 = T.([i / (n+1) for i = 1 : n])
  return RADNLPModel(f, x0, name="genrose_radnlp"; kwargs...)
end

function genrose_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return 1 + 100 * sum((x[i+1] - x[i]^2)^2 for i=1:n-1) + sum((x[i] - 1)^2 for  i=1:n-1)
  end
  x0 = T.([i / (n+1) for i = 1 : n])
  return ADNLPModel(f, x0, name="genrose_autodiff")
end
