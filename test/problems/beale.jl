function beale_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return  (1.5 + x[1] * (1.0 - x[2]))^2 + (2.25 + x[1] * (1.0 - x[2]^2))^2 + (2.625 + x[1] * (1.0 - x[2]^3))^2
  end
  x0 = ones(T, n)
  return RADNLPModel(f, x0, name="beale_radnlp"; kwargs...)
end

function beale_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return  (1.5 + x[1] * (1.0 - x[2]))^2 + (2.25 + x[1] * (1.0 - x[2]^2))^2 + (2.625 + x[1] * (1.0 - x[2]^3))^2
  end
  x0 = ones(T, n)
  return ADNLPModel(f, x0, name="beale_autodiff")
end
