function tointgss_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 3 || error("tointgss : n ≥ 3")
  function f(x)
    n = length(x)
    return sum((10.0 / (n + 2) + x[i+2]^2) * (2.0 - exp(-(x[i] - x[i+1])^2 / (0.1 + x[i+2]^2))) for i=1:n-2)
  end
  x0 = ones(T, n)
  return RADNLPModel(f, x0, name="tointgss_radnlp"; kwargs...)
end

function tointgss_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 3 || error("tointgss : n ≥ 3")
  function f(x)
    n = length(x)
    return sum((10.0 / (n + 2) + x[i+2]^2) * (2.0 - exp(-(x[i] - x[i+1])^2 / (0.1 + x[i+2]^2))) for i=1:n-2)
  end
  x0 = ones(T, n)
  return ADNLPModel(f, x0, name="tointgss_autodiff")
end
