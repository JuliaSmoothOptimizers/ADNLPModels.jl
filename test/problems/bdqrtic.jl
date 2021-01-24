function bdqrtic_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 5 || error("bdqrtic : n ≥ 5")
  function f(x)
    n = length(x)
    return  sum((3 - 4 * x[i])^2 + (x[i]^2 + 2 * x[i+1]^2 + 3 * x[i+2]^2 + 4 * x[i+3]^2 + 5 * x[n]^2)^2 for i=1:n-4)
  end
  x0 = ones(T, n)
  return RADNLPModel(f, x0, name="bdqrtic_radnlp"; kwargs...)
end

function bdqrtic_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 5 || error("bdqrtic : n ≥ 5")
  function f(x)
    n = length(x)
    return  sum((3 - 4 * x[i])^2 + (x[i]^2 + 2 * x[i+1]^2 + 3 * x[i+2]^2 + 4 * x[i+3]^2 + 5 * x[n]^2)^2 for i=1:n-4)
  end
  x0 = ones(T, n)
  return ADNLPModel(f, x0, name="bdqrtic_autodiff")
end
