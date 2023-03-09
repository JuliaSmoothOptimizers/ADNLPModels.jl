module ADNLPModels

# stdlib
using LinearAlgebra, SparseArrays
# external
using ForwardDiff, ReverseDiff, SparseDiffTools, Symbolics
# JSO
using NLPModels
using Requires

abstract type AbstractADNLPModel{T, S} <: AbstractNLPModel{T, S} end
abstract type AbstractADNLSModel{T, S} <: AbstractNLSModel{T, S} end

const ADModel{T, S} = Union{AbstractADNLPModel{T, S}, AbstractADNLSModel{T, S}}

include("ad.jl")
include("sparse_derivatives.jl")
include("forward.jl")
include("reverse.jl")
include("zygote.jl")
include("nlp.jl")
include("nls.jl")

export get_adbackend, set_adbackend!

"""
    get_c(nlp)
    get_c(nlp, ::ADBackend)

Return the out-of-place version of `nlp.c!`.
"""
function get_c(nlp::ADModel)
  function c(x; nnln = nlp.meta.nnln)
    c = similar(x, nnln)
    nlp.c!(c, x)
    return c
  end
  return c
end
get_c(nlp::ADModel, ::ADBackend) = get_c(nlp)
get_c(nlp::ADModel, ::InPlaceADbackend) = nlp.c!

"""
    get_F(nls)
    get_F(nls, ::ADBackend)

Return the out-of-place version of `nls.F!`.
"""
function get_F(nls::AbstractADNLSModel)
  function F(x; nequ = nls.nls_meta.nequ)
    Fx = similar(x, nequ)
    nls.F!(Fx, x)
    return Fx
  end
  return F
end
get_F(nls::AbstractADNLSModel, ::ADBackend) = get_F(nls)

"""
    get_adbackend(nlp)

Returns the value `adbackend` from nlp.
"""
get_adbackend(nlp::ADModel) = nlp.adbackend

"""
    set_adbackend!(nlp, new_adbackend)
    set_adbackend!(nlp; kwargs...)

Replace the current `adbackend` value of nlp by `new_adbackend` or instantiate a new one with `kwargs`, see `ADModelBackend`.
By default, the setter with kwargs will reuse existing backends.
"""
function set_adbackend!(nlp::ADModel, new_adbackend::ADModelBackend)
  nlp.adbackend = new_adbackend
  return nlp
end
function set_adbackend!(nlp::ADModel; kwargs...)
  args = []
  for field in fieldnames(ADNLPModels.ADModelBackend)
    push!(args, if field in keys(kwargs) && typeof(kwargs[field]) <: ADBackend
      kwargs[field]
    elseif field in keys(kwargs) && typeof(kwargs[field]) <: DataType
      if typeof(nlp) <: ADNLPModel
        kwargs[field](nlp.meta.nvar, nlp.f, nlp.meta.ncon; kwargs...)
      elseif typeof(nlp) <: ADNLSModel
        kwargs[field](nlp.meta.nvar, x -> sum(nlp.F(x) .^ 2), nlp.meta.ncon; kwargs...)
      end
    else
      getfield(nlp.adbackend, field)
    end)
  end
  nlp.adbackend = ADModelBackend(args...)
  return nlp
end

end # module
