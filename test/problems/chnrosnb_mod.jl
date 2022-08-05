function chnrosnb_mod_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 2 || ("chnrosnb : n ≥ 2")
  function f(x)
    n = length(x)
    return 16 * sum((x[i-1] - x[i]^2)^2*(1.5+sin(i))^2 for i=2:n) + sum((1.0 - x[i])^2 for i=2:n)
  end
  x0 = -ones(T, n)
  return RADNLPModel(f, x0, name="chnrosnb_radnlp"; kwargs...)
end

function chnrosnb_mod_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 2 || ("chnrosnb : n ≥ 2")
  function f(x)
    n = length(x)
    return 16 * sum((x[i-1] - x[i]^2)^2*(1.5+sin(i))^2 for i=2:n) + sum((1.0 - x[i])^2 for i=2:n)
  end
  x0 = -ones(T, n)
  return ADNLPModel(f, x0, name="chnrosnb_autodiff")
end
