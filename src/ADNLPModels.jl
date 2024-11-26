module ADNLPModels

# stdlib
using LinearAlgebra, SparseArrays

# external
using ADTypes: ADTypes, AbstractColoringAlgorithm, AbstractSparsityDetector
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings
using ForwardDiff, ReverseDiff, Enzyme

# JSO
using NLPModels
using Requires

abstract type AbstractADNLPModel{T, S} <: AbstractNLPModel{T, S} end
abstract type AbstractADNLSModel{T, S} <: AbstractNLSModel{T, S} end

const ADModel{T, S} = Union{AbstractADNLPModel{T, S}, AbstractADNLSModel{T, S}}

include("ad.jl")
include("ad_api.jl")

include("sparsity_pattern.jl")
include("sparse_jacobian.jl")
include("sparse_hessian.jl")

include("forward.jl")
include("reverse.jl")
include("enzyme.jl")
include("zygote.jl")
include("predefined_backend.jl")
include("nlp.jl")

function ADNLPModel!(model::AbstractNLPModel; kwargs...)
  return if model.meta.nlin > 0
    ADNLPModel!(
      x -> obj(model, x),
      model.meta.x0,
      model.meta.lvar,
      model.meta.uvar,
      jac_lin(model, model.meta.x0),
      (cx, x) -> cons!(model, x, cx),
      model.meta.lcon,
      model.meta.ucon;
      kwargs...,
    )
  else
    ADNLPModel!(
      x -> obj(model, x),
      model.meta.x0,
      model.meta.lvar,
      model.meta.uvar,
      (cx, x) -> cons!(model, x, cx),
      model.meta.lcon,
      model.meta.ucon;
      kwargs...,
    )
  end
end

function ADNLPModel(model::AbstractNLPModel; kwargs...)
  function model_c(x; model = model)
    cx = similar(x, model.meta.ncon)
    return cons!(model, x, cx)
  end

  return if model.meta.nlin > 0
    ADNLPModel(
      x -> obj(model, x),
      model.meta.x0,
      model.meta.lvar,
      model.meta.uvar,
      jac_lin(model, model.meta.x0),
      model_c,
      model.meta.lcon,
      model.meta.ucon;
      kwargs...,
    )
  else
    ADNLPModel(
      x -> obj(model, x),
      model.meta.x0,
      model.meta.lvar,
      model.meta.uvar,
      model_c,
      model.meta.lcon,
      model.meta.ucon;
      kwargs...,
    )
  end
end

include("nls.jl")

function ADNLSModel(model::AbstractNLSModel; kwargs...)
  function model_c(x; model = model)
    cx = similar(x, model.meta.ncon)
    return cons!(model, x, cx)
  end
  function model_F(x; model = model)
    Fx = similar(x, model.nls_meta.nequ)
    return residual!(model, x, Fx)
  end

  return if model.meta.nlin > 0
    ADNLSModel(
      model_F,
      model.meta.x0,
      model.nls_meta.nequ,
      model.meta.lvar,
      model.meta.uvar,
      jac_lin(model, model.meta.x0),
      model_c,
      model.meta.lcon,
      model.meta.ucon;
      kwargs...,
    )
  else
    ADNLSModel(
      model_F,
      model.meta.x0,
      model.nls_meta.nequ,
      model.meta.lvar,
      model.meta.uvar,
      model_c,
      model.meta.lcon,
      model.meta.ucon;
      kwargs...,
    )
  end
end

function ADNLSModel!(model::AbstractNLSModel; kwargs...)
  return if model.meta.nlin > 0
    ADNLSModel!(
      (Fx, x) -> residual!(model, x, Fx),
      model.meta.x0,
      model.nls_meta.nequ,
      model.meta.lvar,
      model.meta.uvar,
      jac_lin(model, model.meta.x0),
      (cx, x) -> cons!(model, x, cx),
      model.meta.lcon,
      model.meta.ucon;
      kwargs...,
    )
  else
    ADNLSModel!(
      (Fx, x) -> residual!(model, x, Fx),
      model.meta.x0,
      model.nls_meta.nequ,
      model.meta.lvar,
      model.meta.uvar,
      (cx, x) -> cons!(model, x, cx),
      model.meta.lcon,
      model.meta.ucon;
      kwargs...,
    )
  end
end

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
get_c(::AbstractNLPModel, ::AbstractNLPModel) = () -> ()

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
get_F(nls::AbstractADNLSModel, ::InPlaceADbackend) = nls.F!
get_F(::AbstractNLPModel, ::AbstractNLPModel) = () -> ()

"""
    get_lag(nlp, b::ADBackend, obj_weight)
    get_lag(nlp, b::ADBackend, obj_weight, y)

Return the lagrangian function `ℓ(x) = obj_weight * f(x) + c(x)ᵀy`.
"""
function get_lag(nlp::AbstractADNLPModel, b::ADBackend, obj_weight::Real)
  return ℓ(x; obj_weight = obj_weight) = obj_weight * nlp.f(x)
end

function get_lag(nlp::AbstractADNLPModel, b::ADBackend, obj_weight::Real, y::AbstractVector)
  if nlp.meta.nnln == 0
    return get_lag(nlp, b, obj_weight)
  end
  c = get_c(nlp, b)
  yview = (length(y) == nlp.meta.nnln) ? y : view(y, (nlp.meta.nlin + 1):(nlp.meta.ncon))
  ℓ(x; obj_weight = obj_weight, y = yview) = obj_weight * nlp.f(x) + dot(c(x), y)
  return ℓ
end

function get_lag(nls::AbstractADNLSModel, b::ADBackend, obj_weight::Real)
  F = get_F(nls, b)
  ℓ(x; obj_weight = obj_weight) = obj_weight * mapreduce(Fi -> Fi^2, +, F(x)) / 2
  return ℓ
end
function get_lag(nls::AbstractADNLSModel, b::ADBackend, obj_weight::Real, y::AbstractVector)
  if nls.meta.nnln == 0
    return get_lag(nls, b, obj_weight)
  end
  F = get_F(nls, b)
  c = get_c(nls, b)
  yview = (length(y) == nls.meta.nnln) ? y : view(y, (nls.meta.nlin + 1):(nls.meta.ncon))
  ℓ(x; obj_weight = obj_weight, y = yview) = obj_weight * sum(F(x) .^ 2) / 2 + dot(c(x), y)
  return ℓ
end

get_lag(::AbstractNLPModel, ::AbstractNLPModel, args...) = () -> ()

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
