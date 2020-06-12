# __precompile__()

module ADNLPModels

using LinearAlgebra

using NLPModels

# using FastClosures
using ForwardDiff
using ReverseDiff

export RADNLPModel

# abstract type for models with automatic differentiation
abstract type AbstractADNLPModel <: AbstractNLPModel; end

mutable struct RADNLPModel <: AbstractADNLPModel
  meta :: NLPModels.NLPModelMeta
  counters :: NLPModels.Counters
  f
  ∇f!
  ∇²fprod!
end


function RADNLPModel(f, x0::AbstractVector; name::String="GenericADNLPModel")
  nvar = length(x0)
  meta = NLPModelMeta(nvar, x0=x0, name=name)
  counters = Counters()

  # build in-place objective gradient
  # v = similar(x0)
  # global k = -1; filler() = (k = -k; k); fill!(v, filler())
  # f_tape = ReverseDiff.GradientTape(f, v)
  # compiled_f_tape = ReverseDiff.compile(f_tape)
  # ∇f!(g, x) = ReverseDiff.gradient!(g, compiled_f_tape, x)
  ∇f!(g, x) = ReverseDiff.gradient!(g, f, x)

  # build v → ∇²f(x)v
  # NB: none of the options below works with compiled tapes

  function ∇²fprod!(x::AbstractVector, v::AbstractVector, Hv::AbstractVector)
    z = map(ForwardDiff.Dual, x, v)  # x + ε * v
    ∇fz = similar(z)
    ∇f!(∇fz, z)                      # ∇f(x + ε * v) = ∇f(x) + ε * ∇²f(x)v
    Hv = ForwardDiff.extract_derivative!(Nothing, Hv, ∇fz)  # ∇²f(x)v
    return Hv
  end

  # compute Hessian-vector product ∇²f(x)v by differentiating ϕ(x) := ∇f(x)ᵀv
  # this works but allocates a vector at each eval of ϕ(x)
  # ∇²fprod!(x, v, hv) = ReverseDiff.gradient!(hv,
  #                                         x -> begin
  #                                           g = similar(x)
  #                                           return dot(∇f!(g, x), v)
  #                                         end,
  #                                         x)

  # ∇²fprod!(x, v, hv) = begin
  #   g = ReverseDiff.track.(zero(x))
  #   ReverseDiff.gradient!(hv, x -> dot(∇f!(g, x), v), x)
  # end

  return RADNLPModel(meta, counters, f, ∇f!, ∇²fprod!)
end


function NLPModels.obj(model::RADNLPModel, x::AbstractVector)
  model.counters.neval_obj += 1
  return model.f(x)
end


function NLPModels.grad!(model::RADNLPModel, x::AbstractVector, g::AbstractVector)
  model.counters.neval_grad += 1
  return model.∇f!(g, x)
end


function NLPModels.hprod!(model::RADNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; kwargs...)
  model.counters.neval_hprod += 1
  return model.∇²fprod!(x, v, Hv)
end


function jacvec!(model::RADNLPModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
  z = map(ForwardDiff.Dual, x, v)  # z = x + ε * v
  fz = f(z)  # f(x + εv) = f(x) + ε * J(x) * v
  jv = ForwardDiff.extract_derivative!(Nothing, jv, fz)  # Jf(x) * y
  return Ap
end

end
