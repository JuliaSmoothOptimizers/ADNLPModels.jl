function genrose_nash_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 2 || error("genrose_nash : n ≥ 2")
  function f(x)
    n = length(x)
    return 1.0 + 100 * sum((x[i] - x[i-1]^2)^2 for i=2:n) + sum((1.0 - x[i])^2 for i=2:n)
  end
  x0 = T.([(i/(n+1.0)) for i=1:n])
  return RADNLPModel(f, x0, name="genrose_nash_radnlp"; kwargs...)
end

function genrose_nash_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 2 || error("genrose_nash : n ≥ 2")
  function f(x)
    n = length(x)
    return 1.0 + 100 * sum((x[i] - x[i-1]^2)^2 for i=2:n) + sum((1.0 - x[i])^2 for i=2:n)
  end
  x0 = T.([(i/(n+1.0)) for i=1:n])
  return ADNLPModel(f, x0, name="genrose_nash_autodiff")
end
