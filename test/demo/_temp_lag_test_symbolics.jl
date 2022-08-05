#test:

using Symbolics, LinearAlgebra, SparseArrays

rosenbrock(X) = sum(1:length(X)-1) do i
           100 * (X[i+1] - X[i]^2)^2 + (1 - X[i])^2
end

n=2
@variables xs[1:n]
_roz = rosenbrock(xs)
Symbolics.hessian_sparsity(_roz, xs)
tril(Symbolics.hessian_sparsity(_roz, xs))

sp_hess = Symbolics.sparsehessian(_roz, xs)

Jxs = Symbolics.sparsejacobian([rosenbrock(xs)], xs)

obj_weight = 1.0
m=1
@variables ys[1:m]
_lag = obj_weight * rosenbrock(xs) + dot(ys, [rosenbrock(xs)])
_hess = Symbolics.sparsehessian(_lag, xs)
#=
_temp = tril(sp_hess)
_fun = eval(build_function(_temp, xs, expression = Val{false})[1])
@show _fun(ones(2))
_fun = eval(build_function(_temp, xs, expression = Val{false})[2])
dx = similar(ones(2,2))
_fun(dx, ones(2))
@show dx
=#
