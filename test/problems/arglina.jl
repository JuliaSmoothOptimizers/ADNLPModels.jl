function arglina_radnlp(; n :: Int = 4, type :: Val{T} = Val(Float64)) where T

  m = 2 * n
  x0 = zeros(T, n)
  
  f(x) = sum((x[i] - 2/m * sum(x[j] for j = 1:n) - 1)^2 for i = 1:n) + sum((-2/m * sum(x[j] for j = 1:n) - 1)^2 for i = n+1:m)

  return RADNLPModel(f, x0, name="arglina_radnlp")
end

function arglina_autodiff(; n :: Int = 4, type :: Val{T} = Val(Float64)) where T

  m = 2 * n
  x0 = zeros(T, n)
  
  f(x) = sum((x[i] - 2/m * sum(x[j] for j = 1:n) - 1)^2 for i = 1:n) + sum((-2/m * sum(x[j] for j = 1:n) - 1)^2 for i = n+1:m)

  return ADNLPModel(f, x0, name="arglina_autodiff")
end