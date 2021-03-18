using ADNLPModels: increment!

#Problem 5 in the Hock-Schittkowski suite
function hs5_radnlp(; n :: Int = 4, type::Val{T}=Val(Float64), kwargs...) where T

  x0 = zeros(T, 2)
  f(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 3x[1] / 2 + 5x[2] / 2 + 1
  l = convert(Array{T}, [-1.5; -3.0])
  u = convert(Array{T}, [4.0; 3.0])

  return RADNLPModel(f, x0, l, u, name="hs5_radnlp"; kwargs...)
end
