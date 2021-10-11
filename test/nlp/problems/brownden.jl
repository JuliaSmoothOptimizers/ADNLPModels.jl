export brownden_autodiff

function brownden_autodiff(::Type{T} = Float64) where {T}
  x0 = T[25.0; 5.0; -5.0; -1.0]
  f(x) = begin
    s = zero(T)
    for i = 1:20
      s +=
        (
          (x[1] + x[2] * T(i) / 5 - exp(T(i) / 5))^2 +
          (x[3] + x[4] * sin(T(i) / 5) - cos(T(i) / 5))^2
        )^2
    end
    return s
  end

  return ADNLPModel(f, x0, name = "brownden_autodiff")
end
