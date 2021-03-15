function eg2_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    sum(sin(x[1] + x[i]^2 - 1) for i=1:n-1) + sin(x[n]^2) / 2
  end
  x0 = zeros(T, n)
  return RADNLPModel(f, x0, name="eg2_radnlp"; kwargs...)
end

function eg2_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    sum(sin(x[1] + x[i]^2 - 1) for i=1:n-1) + sin(x[n]^2) / 2
  end
  x0 = zeros(T, n)
  return ADNLPModel(f, x0, name="eg2_autodiff")
end
