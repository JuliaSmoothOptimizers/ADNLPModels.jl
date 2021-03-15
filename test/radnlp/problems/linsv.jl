function linsv_radnlp(;kwargs...)
  
  x0 = zeros(2)
  f(x) = x[1]
  function c(dx, x)
    dx[1] = x[1] + x[2]
    dx[2] = x[2]
    dx
  end
  lcon = [3; 1]
  ucon = [Inf; Inf]
     
  return RADNLPModel(f, x0, c, lcon, ucon, name="linsv_radnlp")
end
