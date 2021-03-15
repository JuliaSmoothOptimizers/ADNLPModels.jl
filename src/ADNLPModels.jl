module ADNLPModels

# stdlib
using LinearAlgebra, SparseArrays
#Forward AD
using ForwardDiff
#Reverse AD
using ReverseDiff, Zygote
#Sparse AD
using SparsityDetection, SparseDiffTools
#Sparse AD
using Symbolics
# JSO
using NLPModels

"""
Abstract type for NLPModels with derivatives computed by automatic differentiation.
"""
abstract type AbstractADNLPModel end

#Import NLPModels functions surcharged by the ADNLPModel
using NLPModels: increment!, decrement!, @lencheck, NLPModelMeta, Counters
import NLPModels: obj, grad, grad!, hess, cons, cons!, jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!, hprod, hprod!, hess_op, hess_op!, jac_structure, jac_structure!, jac_coord, jac_coord!, hess_structure, hess_structure!, hess_coord, hess_coord!

include("model.jl")
include("nlp.jl")
include("nls.jl")

end
