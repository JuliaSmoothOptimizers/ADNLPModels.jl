function cosine_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return sum(cos(x[i]^2 - x[i+1] / 2) for i = 1:n-1)
  end
  x0 = ones(T, n)
  return RADNLPModel(f, x0, name="cosine_radnlp"; kwargs...)
end

function cosine_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return sum(cos(x[i]^2 - x[i+1] / 2) for i = 1:n-1)
  end
  x0 = ones(T, n)
  return ADNLPModel(f, x0, name="cosine_autodiff")
end
