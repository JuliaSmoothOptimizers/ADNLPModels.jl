function sparsqur_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  n ≥ 10 || error("sparsqur : n ≥ 10")
  function f(x)
    n = length(x)
    return 1/8 * sum(
        i * (x[i]^2 +
        x[mod(2*i-1, n) + 1]^2 +
        x[mod(3*i-1, n) + 1]^2 +
        x[mod(5*i-1, n) + 1]^2 +
        x[mod(7*i-1, n) + 1]^2 +
        x[mod(11*i-1, n) + 1]^2)^2 for i=1:n)
  end
  x0 = ones(T, n) / 2
  return RADNLPModel(f, x0, name="sparsqur_radnlp"; kwargs...)
end

function sparsqur_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n ≥ 10 || error("sparsqur : n ≥ 10")
  function f(x)
    n = length(x)
    return 1/8 * sum(
        i * (x[i]^2 +
        x[mod(2*i-1, n) + 1]^2 +
        x[mod(3*i-1, n) + 1]^2 +
        x[mod(5*i-1, n) + 1]^2 +
        x[mod(7*i-1, n) + 1]^2 +
        x[mod(11*i-1, n) + 1]^2)^2 for i=1:n)
  end
  x0 = ones(T, n) / 2
  return ADNLPModel(f, x0, name="sparsqur_autodiff")
end
