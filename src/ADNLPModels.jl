module ADNLPModels

using LinearAlgebra

using NLPModels

using ForwardDiff
using ReverseDiff
using SparsityDetection
using SparseDiffTools
using SparseArrays

export RADNLPModel

mutable struct RADNLPModel <: AbstractADNLPModel
  meta :: NLPModels.NLPModelMeta
  counters :: NLPModels.Counters
  # Functions
  f
  c
end

show_header(io :: IO, model :: RADNLPModel) = println(io, "RADNLPModel - Model with automatic differentiation")



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

  # function jacvec!(model::RADNLPModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
  #   z = map(ForwardDiff.Dual, x, v)  # z = x + ε * v
  #   fz = f(z)  # f(x + εv) = f(x) + ε * J(x) * v
  #   jv = ForwardDiff.extract_derivative!(Nothing, jv, fz)  # Jf(x) * y
  #   return Ap
  # end

  return RADNLPModel(meta, counters, f, ∇f!, ∇²fprod!)
end

function NLPModels.obj(model :: RADNLPModel, x :: AbstractVector)
  increment!(model, :neval_obj)
  return model.f(x)
end

function NLPModels.grad!(model :: RADNLPModel, x :: AbstractVector, g :: AbstractVector)
  increment!(model, :neval_grad)
  ...
  return 
end

function NLPModels.cons!(model :: RADNLPModel, x :: AbstractVector, c :: AbstractVector)
  increment!(model, :neval_cons)
  return c
end

function NLPModels.jac_structure!(model :: RADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  y = similar(model.meta.x0)
  J = jacobian_sparsity(c, y, model.meta.x0)
  rows, cols, _ = findnz(J)
  ...
  return rows, cols
end

function NLPModels.jac_coord!(model :: RADNLPModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(model, :neval_jac)
  ...
  return vals
end

function NLPModels.jprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(model, :neval_jprod)
  ...
  return Jv
end

function NLPModels.jtprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(model, :neval_jtprod)
  ...
  return Jtv
end

function NLPModels.hess_structure!(model :: RADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  H = hessian_sparsity(f, model.meta.x0)
  rows, cols, _ = findnz(H)
  return rows, cols
end

function NLPModels.hess_coord!(model :: RADNLPModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(model, :neval_hess)
  ...
  return vals
end

function NLPModels.hess_coord!(model :: RADNLPModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(model, :neval_hess)
  ...
  return vals
end

function NLPModels.hprod!(model :: RADNLPModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, hv :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(model, :neval_hprod)
  ...
  return hv
end

function NLPModels.hprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, hv :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(model, :neval_hprod)
  ...
  return hv
end

end
