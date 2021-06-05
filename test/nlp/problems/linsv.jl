export linsv_autodiff

function linsv_autodiff()
  x0 = zeros(2)
  f(x) = x[1]
  con(x) = [x[1] + x[2]; x[2]]
  lcon = [3.0; 1.0]
  ucon = [Inf; Inf]

  return ADNLPModel(f, x0, con, lcon, ucon, name = "linsv_autodiff")
end
