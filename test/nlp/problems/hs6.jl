export hs6_autodiff

function hs6_autodiff()
  x0 = [-1.2; 1.0]
  f(x) = (1 - x[1])^2
  c(x) = [10 * (x[2] - x[1]^2)]
  lcon = [0.0]
  ucon = [0.0]

  return ADNLPModel(f, x0, c, lcon, ucon, name = "hs6_autodiff")
end
