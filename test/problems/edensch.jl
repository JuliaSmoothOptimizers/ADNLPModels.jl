function edensch_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return 16 + sum((x[i] - 2)^4 + (x[i] * x[i+1] - 2 * x[i+1])^2 + (x[i+1] + 1)^2 for i=1:n-1)
  end
  x0 = zeros(T, n)
  return RADNLPModel(f, x0, name="edensch_radnlp"; kwargs...)
end

function edensch_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return 16 + sum((x[i] - 2)^4 + (x[i] * x[i+1] - 2 * x[i+1])^2 + (x[i+1] + 1)^2 for i=1:n-1)
  end
  x0 = zeros(T, n)
  return ADNLPModel(f, x0, name="edensch_autodiff")
end
