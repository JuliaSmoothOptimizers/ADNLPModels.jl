function dqrtic_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return sum((x[i] - i)^4 for i=1:n)
  end
  x0 = 2 * ones(T, n)
  return RADNLPModel(f, x0, name="dqrtic_radnlp"; kwargs...)
end

function dqrtic_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return sum((x[i] - i)^4 for i=1:n)
  end
  x0 = 2 * ones(T, n)
  return ADNLPModel(f, x0, name="dqrtic_autodiff")
end
