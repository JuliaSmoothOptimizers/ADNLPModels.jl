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

#in the main code of sparsehessian we have
#=
function sparsehessian(O, vars::AbstractVector; simplify=false)
    O = value(O)
    vars = map(value, vars)
    S = hessian_sparsity(O, vars)
    I, J, _ = findnz(S)
    exprs = Array{Num}(undef, length(I))
    fill!(exprs, 0)
    prev_j = 0
    d = nothing
    for (k, (i, j)) in enumerate(zip(I, J))
        j > i && continue
        if j != prev_j
            d = expand_derivatives(Differential(vars[j])(O), false)
        end
        expr = expand_derivatives(Differential(vars[i])(d), simplify)
        exprs[k] = expr
        prev_j = j
    end
    H = sparse(I, J, exprs, length(vars), length(vars))
    for (i, j) in zip(I, J)
        j > i && (H[i, j] = H[j, i])
    end
    return H
end
####################
# Tanj:
# So, we just need:
#Make a PR, add as an option just the lower triangular ?
#https://github.com/JuliaSymbolics/Symbolics.jl/blob/master/src/diff.jl
function sparsehessian(O, vars::AbstractVector; simplify=false)
    O = value(O)
    vars = map(value, vars)
    S = hessian_sparsity(O, vars)
    I, J, _ = findnz(S)
    exprs = Array{Num}(undef, length(I))
    fill!(exprs, 0)
    prev_j = 0
    d = nothing
    for (k, (i, j)) in enumerate(zip(I, J))
        j > i && continue
        if j != prev_j
            d = expand_derivatives(Differential(vars[j])(O), false)
        end
        expr = expand_derivatives(Differential(vars[i])(d), simplify)
        exprs[k] = expr
        prev_j = j
    end
    H = sparse(I, J, exprs, length(vars), length(vars))
    return H
end
=#

_temp = tril(sp_hess)
_fun = eval(build_function(_temp, xs, expression = Val{false})[1])
@show _fun(ones(2))
_fun = eval(build_function(_temp, xs, expression = Val{false})[2])
dx = similar(ones(2,2))
_fun(dx, ones(2))
@show dx
