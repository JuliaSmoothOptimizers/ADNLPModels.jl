function tquartic_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 2 || error("tquartic : n ≥ 2")
  function f(x)
    n = length(x)
    return (x[1] - 1.0)^2 + sum((x[1]^2 - x[i+1]^2)^2 for i=1:n-2)
  end
  x0 = ones(T, n)
  return RADNLPModel(f, x0, name="tquartic_radnlp"; kwargs...)
end

function tquartic_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 2 || error("tquartic : n ≥ 2")
  function f(x)
    n = length(x)
    return (x[1] - 1.0)^2 + sum((x[1]^2 - x[i+1]^2)^2 for i=1:n-2)
  end
  x0 = ones(T, n)
  return ADNLPModel(f, x0, name="tquartic_autodiff")
end
