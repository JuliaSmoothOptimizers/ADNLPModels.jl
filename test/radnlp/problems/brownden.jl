# Brown and Dennis functions
#
#   Source: Problem 16 in
#   J.J. Moré, B.S. Garbow and K.E. Hillstrom,
#   "Testing Unconstrained Optimization Software",
#   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981
#
#   classification SUR2-AN-4-0
function brownden_radnlp(; n :: Int = 4, type::Val{T}=Val(Float64), kwargs...) where T

  x0 = convert(Array{T}, [25; 5; -5; -1])
  f(x) = begin
    s = zero(T)
    for i = 1:20
      s += ((x[1] + x[2] * T(i)/5 - exp(T(i)/5))^2 + (x[3] + x[4] * sin(T(i)/5) - cos(T(i)/5))^2)^2
    end
    return s
  end

  return RADNLPModel(f, x0, name="brownden_radnlp"; kwargs...)
end
