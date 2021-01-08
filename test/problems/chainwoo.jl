function chainwoo_radnlp(; n :: Int = 4, type :: Val{T} = Val(Float64)) where T

  n % 4 == 0 || error("number of variables must be a multiple of 4")
  x0 = ones(T, n)  
  f(x) = 1.0 + sum(100 * (x[2*i]   - x[2*i-1]^2)^2 + (1 - x[2*i-1])^2 +
              90 * (x[2*i+2] - x[2*i+1]^2)^2 + (1 - x[2*i+1])^2 +
              10 * (x[2*i] + x[2*i+2] - 2)^2 + 0.1 * (x[2*i] - x[2*i+2])^2 for i=1:div(n,2)-1)

  return RADNLPModel(f, x0, name="arwhead_radnlp")
end

function chainwoo_autodiff(; n :: Int = 4, type :: Val{T} = Val(Float64)) where T

  n % 4 == 0 || error("number of variables must be a multiple of 4")
  x0 = ones(T, n)  
  f(x) = 1.0 + sum(100 * (x[2*i]   - x[2*i-1]^2)^2 + (1 - x[2*i-1])^2 +
              90 * (x[2*i+2] - x[2*i+1]^2)^2 + (1 - x[2*i+1])^2 +
              10 * (x[2*i] + x[2*i+2] - 2)^2 + 0.1 * (x[2*i] - x[2*i+2])^2 for i=1:div(n,2)-1)

  return ADNLPModel(f, x0, name="arwhead_autodiff")
end