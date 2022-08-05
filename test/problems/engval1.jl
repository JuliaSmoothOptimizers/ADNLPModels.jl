function engval1_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 2 || error("engval : n ≥ 2")
  function f(x)
    n = length(x)
    return sum(
      (x[i]^2 + x[i+1]^2)^2 - 4 * x[i] + 3
      for i=1:n-1
    )
  end
  x0 = 2 * ones(T, n)
  return RADNLPModel(f, x0, name="engval1_radnlp"; kwargs...)
end

function engval1_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 2 || error("engval : n ≥ 2")
  function f(x)
    n = length(x)
    return sum(
      (x[i]^2 + x[i+1]^2)^2 - 4 * x[i] + 3
      for i=1:n-1
    )
  end
  x0 = 2 * ones(T, n)
  return ADNLPModel(f, x0, name="engval1_autodiff")
end
