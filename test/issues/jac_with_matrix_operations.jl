using SparsityDetection, SparseDiffTools, SparseArrays

#=
Context: implementation of constrained problems

This example illustrates the fact that we cannot call `jacobian_sparsity`
on a function defined with matrix/vector operations.

Q: How do we avoid that?
=#

include("../problems/lincon.jl")

nlp1 = lincon_autodiff()

#=
The following initialization fails due to
ERROR: MethodError: no method matching *(::Zygote.var"#43#44"{var"#c#154"{Array{Int64,2},Array{Int64,1},Array{Int64,2},Array{Int64,2},Array{Int64,1}}}, ::Array{Float64,1})
=#
#nlp2 = lincon_radnlp()

A = [1 2; 3 4]
b = [5; 6]
B = diagm([3 * i for i = 3:5])
c = [1; 2; 3]
C = [0 -2; 4 0]
d = [1; -1]  
 
function c(dx, x)
  dx[1] = 15 * x[15]
  dx[2] = c' * x[10:12]
  dx[3] = d' * x[13:14]
  dx[4] = b' * x[8:9]
  dx[5:6] = C * x[6:7]
  dx[7:8] = A * x[1:2]
  dx[9:11] = B * x[3:5]
  dx
end

x0 = zeros(15)
output = Array{Float64,1}(undef, 11)
s = Sparsity(11, 15)
jacobian_sparsity(c, output, x, sparsity = s, raw = true, verbose = false)