function fletcbv3_mod_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 2 || error("fletcbv3 : n ≥ 2")
  function f(x)
    n = length(x)
    p = 10.0^(-8)
    h = 1.0 / (n + 1)
    return (p / 2.0) * (x[1]^2 + sum((x[i] - x[i+1])^2 for i=1:n-1) + x[n]^2) -
     p * sum(100.0 * (1 + (2.0 / h^2)) * sin(x[i] / 100.0) + (1 / h^2) * cos(x[i]) for i=1:n)
   end
   x0 = T.([(i/(n+1.0)) for i=1:n])
  return RADNLPModel(f, x0, name="fletcbv3_radnlp"; kwargs...)
end

function fletcbv3_mod_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 2 || error("fletcbv3 : n ≥ 2")
  function f(x)
    n = length(x)
    p = 10.0^(-8)
    h = 1.0 / (n + 1)
    return (p / 2.0) * (x[1]^2 + sum((x[i] - x[i+1])^2 for i=1:n-1) + x[n]^2) -
     p * sum(100.0 * (1 + (2.0 / h^2)) * sin(x[i] / 100.0) + (1 / h^2) * cos(x[i]) for i=1:n)
   end
   x0 = T.([(i/(n+1.0)) for i=1:n])
  return ADNLPModel(f, x0, name="fletcbv3_autodiff")
end
