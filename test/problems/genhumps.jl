function genhumps_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    ζ = 20
    return sum((sin(ζ * x[i])^2 * sin(ζ * x[i+1])^2 + (x[i]^2 + x[i+1]^2) / ζ) for i=1:n-1)
  end

  x0 = -506.2 * ones(T, n)
  x0[1] = -506
  return RADNLPModel(f, x0, name="genhumps_radnlp"; kwargs...)
end

function genhumps_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    ζ = 20
    return sum((sin(ζ * x[i])^2 * sin(ζ * x[i+1])^2 + (x[i]^2 + x[i+1]^2) / ζ) for i=1:n-1)
  end

  x0 = -506.2 * ones(T, n)
  x0[1] = -506
  return ADNLPModel(f, x0, name="genhumps_autodiff")
end
