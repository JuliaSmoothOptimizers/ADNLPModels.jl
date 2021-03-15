#=
Context: implementation of constrained problems.

This example illustrates the fact that there are 
no out-of-place call to `jacobian_sparsity`

Q: How do we avoid that?

Comment: right-now RADNLPModel requires in-place constraint function `c(dx, x)`
=#
using SparsityDetection, SparseArrays, SparseDiffTools

#=
function g(x) # out-of-place
  dx = zero(x)
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  dx
end
=#

x0 = [-1.2; 1.0]
c(x) = [10 * (x[2] - x[1]^2)]

function c2(dx, x)
 dx[1] = 10 * (x[2] - x[1]^2)
 dx
end

fin = (dx, x) -> dx .= c(x)

T = Float64
x0 = rand(30)
m, n = 30, 30

#We run (almost) the whole procedure once to get the non-zeros and the config
output = similar(rand(1))
s = Sparsity(m, n)
jacobian_sparsity(c, x0, sparsity = s, raw = true, verbose = false)
S = T.(sparse(s))
colors = matrix_colors(S)
cfH = ForwardColorJacCache(f, x0, colorvec = colors, sparsity = S)
nnzh = nnz(S) 