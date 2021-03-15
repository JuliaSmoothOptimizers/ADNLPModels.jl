function vardim_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return sum((x[i] - 1)^2 for i=1:n) + sum(i * (x[i] - 1) for i=1:n)^2 + sum(i * (x[i] - 1) for i=1:n)^4
  end

  x0 = T.([1 - i/n for i = 1 : n])
  return RADNLPModel(f, x0, name="vardim_radnlp"; kwargs...)
end

function vardim_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return sum((x[i] - 1)^2 for i=1:n) + sum(i * (x[i] - 1) for i=1:n)^2 + sum(i * (x[i] - 1) for i=1:n)^4
  end

  x0 = T.([1 - i/n for i = 1 : n])
  return ADNLPModel(f, x0, name="vardim_autodiff")
end
