#Problem 14 in the Hock-Schittkowski suite
function hs14_radnlp(;kwargs...)
  x0 = [2.0; 2.0]
  f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
  function c(dx, x)
    dx[1] = x[1] - 2 * x[2] + 1
    dx[2] = -x[1]^2/4 - x[2]^2 + 1
    dx
  end
  lcon = [0.0; 0.0]
  ucon = [0.0; Inf]
    
  return RADNLPModel(f, x0, c, lcon, ucon, name="hs14_radnlp")
end
