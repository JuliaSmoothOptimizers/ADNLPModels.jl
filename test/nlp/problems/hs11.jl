export hs11_autodiff

function hs11_autodiff()
  x0 = [4.9; 0.1]
  f(x) = (x[1] - 5)^2 + x[2]^2 - 25
  c(x) = [-x[1]^2 + x[2]]
  lcon = [-Inf]
  ucon = [0.0]

  return ADNLPModel(f, x0, c, lcon, ucon, name = "hs11_autodiff")
end
