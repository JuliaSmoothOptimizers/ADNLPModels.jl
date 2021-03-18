#=
Set of specialized functions from Symbolics adapted to our context
=#

#=
# Return only the lower triangular:
#
#https://github.com/JuliaSymbolics/Symbolics.jl/blob/master/src/diff.jl
#
# @btime tests doesn't really show a difference with sparsehessian.
=#
import Symbolics.sparsehessian
function sparsehessian(O, vars::AbstractVector, tril :: Bool; simplify=false)
    O = Symbolics.value(O)
    vars = map(Symbolics.value, vars)
    S = Symbolics.hessian_sparsity(O, vars)
    I, J, _ = findnz(S)
    exprs = Array{Num}(undef, length(I))
    Symbolics.fill!(exprs, 0)
    prev_j = 0
    d = nothing
    for (k, (i, j)) in enumerate(zip(I, J))
        j > i && continue
        if j != prev_j
            d = Symbolics.expand_derivatives(Symbolics.Differential(vars[j])(O), false)
        end
        expr = Symbolics.expand_derivatives(Symbolics.Differential(vars[i])(d), simplify)
        exprs[k] = expr
        prev_j = j
    end
    H = sparse(I, J, exprs, length(vars), length(vars))
    return H
end

#=
Is it worth having a specialized function for the hess_coord! ?
=#
function sparsehessian_coo(O, vars::AbstractVector; simplify=false)
    O = Symbolics.value(O)
    vars = map(Symbolics.value, vars)
    S = Symbolics.hessian_sparsity(O, vars)
    I, J, _ = findnz(S)
    exprs = Array{Num}(undef, length(I))
    Symbolics.fill!(exprs, 0)
    prev_j = 0
    d = nothing
    for (k, (i, j)) in enumerate(zip(I, J))
        j > i && continue
        if j != prev_j
            d = Symbolics.expand_derivatives(Symbolics.Differential(vars[j])(O), false)
        end
        expr = Symbolics.expand_derivatives(Symbolics.Differential(vars[i])(d), simplify)
        exprs[k] = expr
        prev_j = j
    end
    return (I, J, exprs)
end
