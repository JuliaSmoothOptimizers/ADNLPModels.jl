using NLPModels: increment!

#Problem 10 in the Hock-Schittkowski suite
function hs10_radnlp(;kwargs...)
  x0 = [-10.0; 10.0]
  f(x) = x[1] - x[2]
  function c(dx, x)
    dx[1] = -3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 + 1
    dx
  end
  lcon = [0.0]
  ucon = [Inf]
     
  return RADNLPModel(f, x0, c, lcon, ucon, name="hs10_radnlp")
end
