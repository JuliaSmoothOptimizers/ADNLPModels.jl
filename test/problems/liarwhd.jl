function liarwhd_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 2 || error("liarwhd : n ≥ 2")
  function f(x)
    n = length(x)
    return sum(4.0*(x[i]^2 - x[1])^2 + (x[i] - 1)^2  for i=1:n)
  end
  x0 = ones(T, n)
  return RADNLPModel(f, x0, name="liarwhd_radnlp"; kwargs...)
end

function liarwhd_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 2 || error("liarwhd : n ≥ 2")
  function f(x)
    n = length(x)
    return sum(4.0*(x[i]^2 - x[1])^2 + (x[i] - 1)^2  for i=1:n)
  end
  x0 = ones(T, n)
  return ADNLPModel(f, x0, name="liarwhd_autodiff")
end
