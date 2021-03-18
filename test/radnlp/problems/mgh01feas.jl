export mgh01feas_autodiff

function mgh01feas_radnlp()

  x0 = [-1.2; 1.0]
  f(x) = zero(eltype(x))
  c(x) = [1 - x[1]; 10 * (x[2] - x[1]^2)]
  lcon = zeros(2)
  ucon = zeros(2)

  return ADNLPModel(f, x0, c, lcon, ucon, name="mgh01feas_radnlp")
end
