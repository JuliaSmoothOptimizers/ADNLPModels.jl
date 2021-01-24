function arglinc_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    m=2*n
    return 2 + sum(((i-1) * sum(j * x[j] for j = 2:n-1) - 1)^2 for i = 2:m-1)
  end
  x0 = ones(T, n)
  return RADNLPModel(f, x0, name="arglinc_radnlp"; kwargs...)
end

function arglinc_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    m=2*n
    return 2 + sum(((i-1) * sum(j * x[j] for j = 2:n-1) - 1)^2 for i = 2:m-1)
  end
  x0 = ones(T, n)
  return ADNLPModel(f, x0, name="arglinc_autodiff")
end
