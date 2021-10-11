export nlslc_autodiff

function nlslc_autodiff(::Type{T} = Float64) where {T}
  A = T[1 2; 3 4]
  b = T[5; 6]
  B = diagm(T[3 * i for i = 3:5])
  c = T[1; 2; 3]
  C = T[0 -2; 4 0]
  d = T[1; -1]

  x0 = zeros(T, 15)
  F(x) = [x[i]^2 - i^2 for i = 1:15]
  con(x) = [
    15 * x[15]
    c' * x[10:12]
    d' * x[13:14]
    b' * x[8:9]
    C * x[6:7]
    A * x[1:2]
    B * x[3:5]
  ]

  lcon = T[22.0; 1.0; -Inf; -11.0; -d; -b; -Inf * ones(3)]
  ucon = T[22.0; Inf; 16.0; 9.0; -d; Inf * ones(2); c]

  return ADNLSModel(F, x0, 15, con, lcon, ucon, name = "nlslincon_autodiff")
end
