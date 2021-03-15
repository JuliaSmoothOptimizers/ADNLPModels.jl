function arglinb_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    m = 2*n
    return sum((i * sum(j * x[j] for j = 1:n) - 1)^2 for i = 1:m)
  end
  x0 = ones(T, n)
  return RADNLPModel(f, x0, name="arglinb_radnlp"; kwargs...)
end

function arglinb_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    m = 2*n
    return sum((i * sum(j * x[j] for j = 1:n) - 1)^2 for i = 1:m)
  end
  x0 = ones(T, n)
  return ADNLPModel(f, x0, name="arglinb_autodiff")
end
