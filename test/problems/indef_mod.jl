function indef_mod_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 3 || error("indef : n ≥ 3")
  function f(x)
    n = length(x)
    return 100.0 * sum(sin(x[i] / 100.0) for i=1:n) + 0.5 * sum(cos(2.0 * x[i] - x[n] - x[1]) for i=2:n-1)
  end
  x0 = T.([(i/(n+1.0)) for i=1:n])
  return RADNLPModel(f, x0, name="indef_radnlp"; kwargs...)
end

function indef_mod_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 3 || error("indef : n ≥ 3")
  function f(x)
    n = length(x)
    return 100.0 * sum(sin(x[i] / 100.0) for i=1:n) + 0.5 * sum(cos(2.0 * x[i] - x[n] - x[1]) for i=2:n-1)
  end
  x0 = T.([(i/(n+1.0)) for i=1:n])
  return ADNLPModel(f, x0, name="indef_autodiff")
end
