export RADNLPModel

"""
    RADNLPModel(f, x0)
    RADNLPModel(f, x0, lvar, uvar)
RADNLPModel is an AbstractNLPModel using automatic differentiation to compute the derivatives.
The problem is defined as
     min  f(x)
    s.to  lcon ≤ c(x) ≤ ucon
          lvar ≤   x  ≤ uvar.
The following keyword arguments are available to all constructors:
- `name`: The name of the model (default: "Generic")
The following keyword arguments are available to the constructors for constrained problems:
- `lin`: An array of indexes of the linear constraints (default: `Int[]`)
- `y0`: An inital estimate to the Lagrangian multipliers (default: zeros)
"""
mutable struct RADNLPModel <: AbstractNLPModel #AbstractADNLPModel
  meta :: NLPModelMeta
  counters :: Counters
  # Functions
  f :: Function
  c :: Function
  ∇f! :: Function
  cfg #config gradient Union{Nothing, ReverseDiff.CompiledTape}
end

show_header(io :: IO, model :: RADNLPModel) = println(io, "RADNLPModel - Model with automatic differentiation")

function RADNLPModel(meta :: AbstractNLPModelMeta,
                     f    :: Function)
  
  counters = Counters()

  # build in-place objective gradient
   v = similar(meta.x0)
   #global k = -1; filler() = (k = -k; k); fill!(v, filler())
   f_tape = ReverseDiff.GradientTape(f, v)
   cfg = ReverseDiff.compile(f_tape) #compiled_f_tape typeof(compiled_f_tape) <: ReverseDiff.CompiledTape
   ∇f!(g, x, cfg) = ReverseDiff.gradient!(g, cfg, x)
  #Option 1
  #∇f!(g, x, cfg) = ReverseDiff.gradient!(g, f, x)
  #cfg = nothing
  #Option 2
  #=
  cfg = nothing
  function ∇f!(g, x, cfg)
    _g = Zygote.gradient(f, x)
    g .= typeof(_g) <: AbstractVector ? _g : _g[1] #see benchmark/bug_zygote.jl
    return g
  end
  =#

  # build v → ∇²f(x)v
  # NB: none of the options below works with compiled tapes

  function ∇²fprod!(x::AbstractVector, v::AbstractVector, Hv::AbstractVector)
    z = map(ForwardDiff.Dual, x, v)  # x + ε * v
    ∇fz = similar(z)
    ∇f!(∇fz, z, nothing)                      # ∇f(x + ε * v) = ∇f(x) + ε * ∇²f(x)v
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

  return RADNLPModel(meta, counters, f, x->T[], ∇f!, cfg) #, ∇²fprod!)
end

function RADNLPModel(f :: Function, x0::AbstractVector{T}; name::String="GenericADNLPModel") where T
  nvar = length(x0)
  @lencheck nvar x0

  @warn "nnzh is not precised here."
  nnzh = nvar * (nvar + 1) / 2
  
  meta = NLPModelMeta(nvar, x0=x0, nnzh=nnzh, minimize=true, islp=false, name=name)
  return RADNLPModel(meta, f)
end

function RADNLPModel(f    :: Function, 
                     x0   :: AbstractVector{T}, 
                     lvar :: AbstractVector, 
                     uvar :: AbstractVector;
                     name :: String = "Generic") where T
                    
  nvar = length(x0)
  @lencheck nvar x0 lvar uvar

  @warn "nnzh is not precised here."
  nnzh = nvar * (nvar + 1) / 2

  meta = NLPModelMeta(nvar, x0 = x0, lvar = lvar, uvar = uvar, nnzh = nnzh, 
                      minimize = true, islp = false, name = name)
  return RADNLPModel(meta, f)
end

function NLPModels.obj(model :: RADNLPModel, x :: AbstractVector)
  @lencheck model.meta.nvar x
  increment!(model, :neval_obj)
  return model.f(x)
end

function NLPModels.grad!(model :: RADNLPModel, x :: AbstractVector, g :: AbstractVector)
  @lencheck model.meta.nvar x g
  increment!(model, :neval_grad)
  model.∇f!(g, x, model.cfg)
  return g
end

function NLPModels.cons!(model :: RADNLPModel, x :: AbstractVector, c :: AbstractVector)
  @lencheck model.meta.nvar x
  @lencheck model.meta.ncon c
  increment!(model, :neval_cons)
  c .= model.c(x)
  return c
end

function NLPModels.jac_structure!(model :: RADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck model.meta.nnzj rows cols
  y = similar(model.meta.x0)
  J = jacobian_sparsity(model.c, y, model.meta.x0)
  rows, cols, _ = findnz(J)
  #...
  return rows, cols
end

function NLPModels.jac_coord!(model :: RADNLPModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck model.meta.nvar x
  @lencheck model.meta.nnzj vals
  increment!(model, :neval_jac)
  #...
  return vals
end

function NLPModels.jprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck model.meta.nvar x v
  @lencheck model.meta.ncon Jv
  increment!(model, :neval_jprod)
  #...
  return Jv
end

function NLPModels.jtprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck model.meta.nvar x Jtv
  @lencheck model.meta.ncon v
  increment!(model, :neval_jtprod)
  #...
  return Jtv
end

function NLPModels.hess_structure!(model :: RADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck model.meta.nnzh rows cols
  H = hessian_sparsity(model.f, model.meta.x0)
  rows, cols, _ = findnz(H)
  return rows, cols
end

function NLPModels.hess_coord!(model :: RADNLPModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x
  @lencheck model.meta.nnzh vals
  increment!(model, :neval_hess)
  #...
  return vals
end

function NLPModels.hess_coord!(model :: RADNLPModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x
  @lencheck model.meta.nnzh vals
  increment!(model, :neval_hess)
  #...
  return vals
end

function NLPModels.hprod!(model :: RADNLPModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x v Hv
  increment!(model, :neval_hprod)
  #...
  return hv
end

function NLPModels.hprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x v Hv
  @lencheck model.meta.ncon y
  increment!(model, :neval_hprod)
  #...
  return hv
end
