function dqdrtic_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return sum(x[i]^2 + 100 * (x[i+1]^2 + x[i+2]^2) for i=1:n-2)
  end
  x0 = 3 * ones(T, n)
  return RADNLPModel(f, x0, name="dqdrtic_radnlp"; kwargs...)
end

function dqdrtic_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return sum(x[i]^2 + 100 * (x[i+1]^2 + x[i+2]^2) for i=1:n-2)
  end
  x0 = 3 * ones(T, n)
  return ADNLPModel(f, x0, name="dqdrtic_autodiff")
end
