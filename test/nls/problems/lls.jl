export lls_autodiff

function lls_autodiff()

  x0 = [0.0; 0.0]
  F(x) = [x[1] - x[2]; x[1] + x[2] - 2; x[2] - 2]
  c(x) = [x[1] + x[2]]
  lcon = [0.0]
  ucon = [Inf]

  return ADNLSModel(F, x0, 3, c, lcon, ucon, name="lls_autodiff")
end
