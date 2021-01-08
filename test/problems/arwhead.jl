function arwhead_radnlp(; n :: Int = 4, type :: Val{T} = Val(Float64)) where T

  x0 = ones(T, n)  
  f(x) = sum((x[i]^2 + x[n]^2)^2 - 4 * x[i] + 3 for i=1:n-1)

  return RADNLPModel(f, x0, name="arwhead_radnlp")
end

function arwhead_autodiff(; n :: Int = 4, type :: Val{T} = Val(Float64)) where T

  x0 = ones(T, n)  
  f(x) = sum((x[i]^2 + x[n]^2)^2 - 4 * x[i] + 3 for i=1:n-1)

  return ADNLPModel(f, x0, name="arwhead_autodiff")
end