function extrosnb_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return 100 * sum((x[i] - x[i-1]^2)^2 for i=2:n) + (1 - x[1])^2
  end
  x0 = -ones(T, n)
  return RADNLPModel(f, x0, name="extrosnb_radnlp"; kwargs...)
end

function extrosnb_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return 100 * sum((x[i] - x[i-1]^2)^2 for i=2:n) + (1 - x[1])^2
  end
  x0 = -ones(T, n)
  return ADNLPModel(f, x0, name="extrosnb_autodiff")
end
