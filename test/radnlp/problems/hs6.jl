using NLPModels: increment!

function hs6_radnlp(;kwargs...)
  x0 = [-1.2; 1.0]
  f(x) = (1 - x[1])^2
  #=
  function c(dx, x)
    dx[1] = 10 * (x[2] - x[1]^2)
    dx
  end
  =#
  c(x) = [10 * (x[2] - x[1]^2)]
  lcon = [0.0]
  ucon = [0.0]
  
  return RADNLPModel(f, x0, c, lcon, ucon, name="hs6_radnlp")
end
