export RADNLPModel

include("symbolics_extra_fcts.jl")

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
  meta     :: NLPModelMeta
  counters :: Counters

  # Functions
  f   :: Function
  c   :: Function
  ∇f! :: Function
  cfg #config gradient Union{Nothing, ReverseDiff.CompiledTape}

  cfH #config for the Hessian computation
  cfJ #config for the Jacobian computation
end

show_header(io :: IO, model :: RADNLPModel) = println(io, "RADNLPModel - Model with automatic differentiation")

"""
Create an RADNLPModel with ReverseDiff.gradient! and precompilation work.
"""
function smart_reverse(x0 :: AbstractVector,
                       f  :: Function)

  # build in-place objective gradient
  v = similar(x0)
  #global k = -1; filler() = (k = -k; k); fill!(v, filler())
  f_tape = ReverseDiff.GradientTape(f, v)
  cfg = ReverseDiff.compile(f_tape) #compiled_f_tape typeof(compiled_f_tape) <: ReverseDiff.CompiledTape
  ∇f!(g, x, cfg) = ReverseDiff.gradient!(g, cfg, x)

  return (∇f!, cfg)
end

"""
Create an RADNLPModel with ReverseDiff.gradient!
"""
function reverse(:: AbstractVector,
                 f :: Function)

  cfg = nothing
  ∇f!(g, x, cfg) = ReverseDiff.gradient!(g, f, x)

return (∇f!, cfg)
end

#=
function RADNLPModel(meta :: AbstractNLPModelMeta,
                     f    :: Function,
                     cfH  :: ForwardColorJacCache;
                     gradient :: Function = smart_reverse)
  
  counters = Counters()
  #=
  cfg = nothing
  function ∇f!(g, x, cfg)
    _g = Zygote.gradient(f, x)
    g .= typeof(_g) <: AbstractVector ? _g : _g[1] #see benchmark/bug_zygote.jl
    return g
  end
  =#
  (∇f!, cfg) = gradient(meta.x0, f)
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
  return RADNLPModel(meta, counters, f, x->T[], ∇f!, cfg, cfH) #, ∇²fprod!)
end
=#

"""
Given a function `f(x)` of size `m`, and a vector `x0` of size `n`,
returns the number of non-zeros in the jacobian (after forming the sparse matrix),
and the config data.
"""
function _meta_function(c :: Function, x0 :: AbstractVector{T}, m :: Int, n :: Int, ::Val{false}) where T
  
  @variables xs[1:n]
  _fun = c(xs)
  S    = Symbolics.sparsejacobian(_fun, xs)
  cfJ  = Symbolics.build_function(S, xs, expression = Val{false})
  nnzh = nnz(S) 

 return nnzh, cfJ
end

function _meta_function(c :: Function, x0 :: AbstractVector{T}, m :: Int, n :: Int, ::Val{true}) where T

  @variables xs[1:n]
  _fun = c(xs)
  S    = Symbolics.jacobian_sparsity(_fun, xs)
  cfJ  = nothing
  nnzh = nnz(S) 

 return nnzh, cfJ
end

function _meta_function(f :: Function, x0 :: AbstractVector{T}, n :: Int, ::Val{false}) where T

  @variables xs[1:n]
  _fun = f(xs)
  S    = tril(Symbolics.sparsehessian(_fun, xs))
  #use keyword `expression = Val{false}` to get a pre-compiled code
  #pair of out-of-place / in-place function
  cfH  = Symbolics.build_function(S, xs, expression = Val{false})
  nnzh = nnz(S) 

 return nnzh, cfH
end

function _meta_function(f :: Function, x0 :: AbstractVector{T}, n :: Int, ::Val{true}) where T

  @variables xs[1:n]
  _fun = f(xs)
  S = tril(Symbolics.hessian_sparsity(_fun, xs))
  cfH = nothing #factorization-free adaptation
  nnzh = nnz(S)

 return nnzh, cfH
end

function RADNLPModel(f         :: Function, 
                     x0        :: AbstractVector{T}; 
                     name      :: String="GenericADNLPModel", 
                     gradient  :: Function = smart_reverse,
                     hess_free :: Bool = false, 
                     kwargs...) where T
  nvar = length(x0)
  @lencheck nvar x0

  (∇f!, cfg) = gradient(x0, f)
  nnzh, cfH = _meta_function(f, x0, nvar, Val(hess_free))
  
  meta = NLPModelMeta(nvar, x0=x0, nnzh=nnzh, minimize=true, islp=false, name=name)

  counters = Counters()

  return RADNLPModel(meta, counters, f, x->T[], ∇f!, cfg, cfH, nothing)
end

function RADNLPModel(f         :: Function, 
                     x0        :: AbstractVector{T}, 
                     lvar      :: AbstractVector, 
                     uvar      :: AbstractVector;
                     name      :: String = "Generic",
                     gradient  :: Function = smart_reverse,
                     hess_free :: Bool = false,  
                     kwargs...) where T
                    
  nvar = length(x0)
  @lencheck nvar x0 lvar uvar

  (∇f!, cfg) = gradient(x0, f)
  nnzh, cfH = _meta_function(f, x0, nvar, Val(hess_free))

  meta = NLPModelMeta(nvar, x0 = x0, lvar = lvar, uvar = uvar, nnzh = nnzh, 
                      minimize = true, islp = false, name = name)
  
  counters = Counters()
  
  return RADNLPModel(meta, counters, f, x->T[], ∇f!, cfg, cfH, nothing)
end

function RADNLPModel(f         :: Function, 
                     x0        :: AbstractVector{T},
                     c         :: Function,
                     lcon      :: AbstractVector,
                     ucon      :: AbstractVector; 
                     name      :: String="GenericADNLPModel",
                     y0        :: AbstractVector=fill!(similar(lcon), zero(T)),
                     lin       :: AbstractVector{<: Integer}=Int[], 
                     gradient  :: Function = smart_reverse, 
                     hess_free :: Bool = false, 
                     jac_free  :: Bool = false, 
                     kwargs...) where T

  nvar = length(x0)
  ncon = length(lcon)
  @lencheck nvar x0
  @lencheck ncon ucon y0
  
  (∇f!, cfg) = gradient(x0, f)
  nnzh, cfH = _meta_function(f, x0, nvar, Val(hess_free))
  nnzj, cfJ = _meta_function(c, x0, ncon, nvar, Val(jac_free))

  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, ncon=ncon, y0=y0, lcon=lcon, ucon=ucon,
                      nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true,
                      islp=false, name=name)

  counters = Counters()

  return RADNLPModel(meta, counters, f, c, ∇f!, cfg, cfH, cfJ)
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

  @variables xs[1:model.meta.nvar]
  _fun = model.c(xs)
  Jx   = Symbolics.jacobian_sparsity(_fun, xs)

  rows, cols, _ = findnz(Jx) 
  return rows, cols
end

function NLPModels.jac_coord!(model :: RADNLPModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck model.meta.nvar x
  @lencheck model.meta.nnzj vals
  increment!(model, :neval_jac)
  #####################################
  # Tangi: to be improved:
  #ideally we want to use the in-place version but it returns a matrix...
  _fun  = eval(model.cfJ[1])
  Jx    = Base.invokelatest(_fun, x)
  vals .= Jx.nzval
  return vals
end

function NLPModels.jac(model :: RADNLPModel, x :: AbstractVector)
  @lencheck model.meta.nvar x
  increment!(model, :neval_jac)

  if isnothing(model.cfJ) throw(error("This is a matrix-free ADNLPModel.")) end
  _fun = eval(model.cfJ[1])
  Jx   = Base.invokelatest(_fun, x)
  return Jx
end

function NLPModels.jprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck model.meta.nvar x v
  @lencheck model.meta.ncon Jv
  increment!(model, :neval_jprod)
  Jv .= ForwardDiff.derivative(t -> model.c(x + t * v), 0)
  return Jv
end

function NLPModels.jtprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck model.meta.nvar x Jtv
  @lencheck model.meta.ncon v
  increment!(model, :neval_jtprod)
  Jtv .= ForwardDiff.gradient(x -> dot(model.c(x), v), x) #ReverseDiff without preparation isn't better
  return Jtv
end

function NLPModels.hess_structure!(model :: RADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck model.meta.nnzh rows cols
  #using Symbolics.hessian_sparsity
  @variables xs[1:model.meta.nvar]
  _fun = model.f(xs)
  H = tril(Symbolics.hessian_sparsity(_fun, xs)) #with boolean values
  rows, cols, _ = findnz(H) 
  return rows, cols
end

function NLPModels.hess_coord!(model :: RADNLPModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x
  @lencheck model.meta.nnzh vals
  increment!(model, :neval_hess)
  if isnothing(model.cfH) throw(error("This is a matrix-free ADNLPModel.")) end
  
  #####################################
  # Tangi: to be improved:
  _fun = eval(model.cfH[1])
  Hx = Base.invokelatest(_fun, x)
  #_fun = eval(model.cfH[2]) #ideally we want to use the in-place version but it returns a matrix...
  #J = _fun(vals, x)
  vals .= Hx.nzval
  return vals
end

function NLPModels.hess_coord!(model :: RADNLPModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x
  @lencheck model.meta.ncon y
  @lencheck model.meta.nnzh vals
  increment!(model, :neval_hess)
  #... TODO
  return vals
end

function NLPModels.hprod!(model :: RADNLPModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x v Hv
  @lencheck model.meta.ncon y
  increment!(model, :neval_hprod)
  #... TODO
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function NLPModels.hprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x v Hv
  increment!(model, :neval_hprod)
  #...
  #= Option 1
  ℓ(x) = obj_weight * nlp.f(x)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  =#
  #= Option 2
  autoback_hesvec(nlp.f,nlp.meta.x0,nlp.meta.x0)
  =#
  return Hv
end

function hess(model :: RADNLPModel, x :: AbstractVector{T}; obj_weight :: Real = one(T)) where T
  @lencheck model.meta.nvar x
  increment!(model, :neval_hess)
  if model.meta.nnzh == 0
    return spzeros(T, model.meta.nvar, model.meta.nvar)
  elseif obj_weight == 0.
    rows, cols = hess_structure(model)
    return sparse(T, rows, cols, zeros(T, model.meta.nnzh), model.meta.nvar, model.meta.nvar)
  end
  #ℓ(x) = obj_weight * nlp.f(x)
  #Option 1:
  #forwarddiff_color_jacobian(J, x->ForwardDiff.gradient(nlp.f,x), x, nlp.cfH);
  #Option 2:
  #Hx = forwarddiff_color_jacobian(J, x->ReverseDiff.gradient(nlp.f,x), x, nlp.cfH);
  #Option 3:
  # Use the gradient config ? and ReverseDiff
  #Option 0: the old one
  #Hx = obj_weight * ForwardDiff.hessian(nlp.f, x)
  #Option 4: using symbolics:
  if isnothing(model.cfH) throw(error("This is a matrix-free ADNLPModel.")) end
  _fun = eval(model.cfH[1])
  Hx = Base.invokelatest(_fun, x)
  return obj_weight * Hx
end

function hess(nlp :: RADNLPModel, x :: AbstractVector, y :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end
