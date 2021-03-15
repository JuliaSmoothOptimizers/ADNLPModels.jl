#Problem 11 in the Hock-Schittkowski suite
function hs11_radnlp(;kwargs...)
  x0 = [4.9; 0.1]
  f(x) = (x[1] - 5)^2 + x[2]^2 - 25
  #=
  function c(dx, x)
    dx[1] = -x[1]^2 + x[2]
    dx
  end
  =#
  c(x) = [-x[1]^2 + x[2]]
  lcon = [-Inf]
  ucon = [0.0]
   
  return RADNLPModel(f, x0, c, lcon, ucon, name="hs11_radnlp")
end
