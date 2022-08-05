function fletchcr_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return 100 * sum((x[i+1] - x[i] + 1 - x[i]^2)^2 for i=1:n-1)
  end
  x0 = zeros(T, n)
  return RADNLPModel(f, x0, name="fletchcr_radnlp"; kwargs...)
end

function fletchcr_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return 100 * sum((x[i+1] - x[i] + 1 - x[i]^2)^2 for i=1:n-1)
  end
  x0 = zeros(T, n)
  return ADNLPModel(f, x0, name="fletchcr_autodiff")
end
