using Symbolics # version 0.1.4 - 0.1.10
f(x)= (1.5 + x[1] * (1 - x[2]))^2 + (2.25 + x[1] * (1 - x[2]^2))^2 + (2.625 + x[1] * (1 - x[2]^3))^2
@variables xs[1:2]
_fun = f(xs)
S = Symbolics.sparsehessian(_fun, xs)
cfH = Symbolics.build_function(S, xs)

@warn "Tangi: an alternative patch is"
Symbolics.build_function(S.nzval, xs) #and then we would get the proper hess_coord !

#1st step:
#Symbolics._build_function(Symbolics.JuliaTarget(), S, xs)
# l.183: 
#2nd step:

dargs = map(Symbolics.destructure_arg, xs)
i = findfirst(x->x isa Symbolics.DestructuredArgs, dargs)
similarto = i === nothing ? Array : dargs[i].name

#oop_expr = Func(dargs, [], Symbolics.make_array(Symbolics.SerialForm(), dargs, S, similarto))
#3rd step:
#Symbolics._make_array(S, similarto)
#4th step:
arr = map(x->Symbolics._make_array(x, similarto), S) #arr = map(x->Symbolics._make_array(x, similarto), S.nzval)
f = x->Symbolics._make_array(x, similarto)
#=
Bs = similar(S)
SparseArrays._noshapecheck_map(f, S, Bs)
=#

#=
function _noshapecheck_map(f::Tf, A::SparseVecOrMat, Bs::Vararg{SparseVecOrMat,N}) where {Tf,N}
    fofzeros = f(_zeros_eltypes(A, Bs...)...)
    fpreszeros = _iszero(fofzeros)
    maxnnzC = Int(fpreszeros ? min(widelength(A), _sumnnzs(A, Bs...)) : widelength(A))
    entrytypeC = Base.Broadcast.combine_eltypes(f, (A, Bs...))
    indextypeC = _promote_indtype(A, Bs...)
    C = _allocres(size(A), indextypeC, entrytypeC, maxnnzC)
    return fpreszeros ? _map_zeropres!(f, C, A, Bs...) :
                        _map_notzeropres!(f, fofzeros, C, A, Bs...)
end
=#
#=
function _map_zeropres!(f::Tf, C::SparseVecOrMat, A::SparseVecOrMat) where Tf
    spaceC::Int = min(length(storedinds(C)), length(storedvals(C)))
    Ck = 1
    @inbounds for j in columns(C)
        setcolptr!(C, j, Ck)
        for Ak in colrange(A, j)
            Cx = f(storedvals(A)[Ak])
            if !_iszero(Cx)
                Ck > spaceC && (spaceC = expandstorage!(C, Ck + nnz(A) - (Ak - 1)))
                storedinds(C)[Ck] = storedinds(A)[Ak]
                storedvals(C)[Ck] = Cx
                Ck += 1
            end
        end
    end
    @inbounds setcolptr!(C, numcols(C) + 1, Ck)
    trimstorage!(C, Ck - 1)
    return C
end
=#
#=
@inline _iszero(x) = x == 0
=#
#=
function Base.iszero(x::Num)
    _x = SymbolicUtils.to_mpoly(value(x))[1]
    return (_x isa Number || _x isa SymbolicUtils.MPoly) && iszero(_x)
end
=#
