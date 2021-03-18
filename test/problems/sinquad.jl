function sinquad_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return (x[1] - 4)^4 + (x[n]^2 - x[1]^2)^2 + sum((sin(x[i] - x[n]) - x[1]^2 + x[i]^2)^2 for i=2:n-1)
  end
  x0 = ones(T, n) / 10
  return RADNLPModel(f, x0, name="sinquad_radnlp"; kwargs...)
end

function sinquad_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return (x[1] - 4)^4 + (x[n]^2 - x[1]^2)^2 + sum((sin(x[i] - x[n]) - x[1]^2 + x[i]^2)^2 for i=2:n-1)
  end
  x0 = ones(T, n) / 10
  return ADNLPModel(f, x0, name="sinquad_autodiff")
end
