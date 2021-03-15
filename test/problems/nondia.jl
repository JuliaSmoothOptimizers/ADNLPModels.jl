function nondia_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 2 || error("nondia : n ≥ 2")
  function f(x)
    n = length(x)
    return (x[1] - 1.0)^2 + sum((100.0*x[1] - x[i-1]^2)^2 for i=2:n)
  end
  x0 = -ones(T, n)
  return RADNLPModel(f, x0, name="nondia_radnlp"; kwargs...)
end

function nondia_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 2 || error("nondia : n ≥ 2")
  function f(x)
    n = length(x)
    return (x[1] - 1.0)^2 + sum((100.0*x[1] - x[i-1]^2)^2 for i=2:n)
  end
  x0 = -ones(T, n)
  return ADNLPModel(f, x0, name="nondia_autodiff")
end
