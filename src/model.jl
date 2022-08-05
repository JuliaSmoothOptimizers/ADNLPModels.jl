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
  f
  c
  ∇f!
  cfg #config gradient Union{Nothing, ReverseDiff.CompiledTape}
  cfH #config for the Hessian computation
  cfJ #config for the Jacobian computation
  cfℓ #config the Lagrangian Hessian computation
end

show_header(io :: IO, model :: RADNLPModel) = println(io, "RADNLPModel - Model with automatic differentiation")

"""
Create an RADNLPModel with ReverseDiff.gradient! and precompilation work.
"""
function smart_reverse(x0 :: AbstractVector, f)

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
function reverse(:: AbstractVector, f)

  cfg = nothing
  ∇f!(g, x, cfg) = ReverseDiff.gradient!(g, f, x)

return (∇f!, cfg)
end

#=
function RADNLPModel(meta :: AbstractNLPModelMeta,
                     f,
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
function _meta_function(c, x0 :: AbstractVector{T}, m :: Int, n :: Int, ::Val{false}) where T
  
  @variables xs[1:n]
  _fun = c(xs)
  S    = Symbolics.sparsejacobian(_fun, xs)
  cfJ  = Symbolics.build_function(S.nzval, xs, expression = Val{false})
  nnzh = nnz(S) 

 return nnzh, cfJ
end

function _meta_function(c, x0 :: AbstractVector{T}, m :: Int, n :: Int, ::Val{true}) where T

  @variables xs[1:n]
  _fun = c(xs)
  S    = Symbolics.jacobian_sparsity(_fun, xs)
  cfJ  = nothing
  nnzh = nnz(S) 

 return nnzh, cfJ
end

function _meta_function(f, x0 :: AbstractVector{T}, n :: Int, ::Val{false}) where T

  @variables xs[1:n]
  _fun = f(xs)
  S    = tril(Symbolics.sparsehessian(_fun, xs))
  #use keyword `expression = Val{false}` to get a pre-compiled code
  #pair of out-of-place / in-place function
  cfH  = Symbolics.build_function(S.nzval, xs, expression = Val{false})
  nnzh = nnz(S) 

 return nnzh, (cfH[1], cfH[2], nnzh)
end

function _meta_function(f, x0 :: AbstractVector{T}, n :: Int, ::Val{true}) where T

  @variables xs[1:n]
  _fun = f(xs)
  S = tril(Symbolics.hessian_sparsity(_fun, xs))
  cfH = nothing #factorization-free adaptation
  nnzh = nnz(S)

 return nnzh, cfH
end

function _meta_function(f, c, x0 :: AbstractVector{T}, m :: Int, n :: Int, ::Val{true}) where T

  @variables xs[1:n]
  @variables ys[1:m]
  _lag = f(xs) + dot(ys, c(xs))
  S = tril(Symbolics.hessian_sparsity(_lag, xs))
  cfℓ = nothing #factorization-free adaptation
  nnzh = nnz(S)

 return nnzh, cfℓ
end

function _meta_function(f, c, x0 :: AbstractVector{T}, m :: Int, n :: Int, ::Val{false}) where T

  @variables xs[1:n]
  @variables ys[1:m]
  @variables obj_weight
  _lag = obj_weight * f(xs) + dot(ys, c(xs))
  S = tril(Symbolics.sparsehessian(_lag, xs))
  #use keyword `expression = Val{false}` to get a pre-compiled code
  #pair of out-of-place / in-place function
  cfℓ = Symbolics.build_function(S.nzval, xs, ys, obj_weight, expression = Val{false})
  nnzh = nnz(S)

 return nnzh, cfℓ
end

function RADNLPModel(f, 
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

  return RADNLPModel(meta, counters, f, x->T[], ∇f!, cfg, cfH, nothing, nothing)
end

function RADNLPModel(f, 
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
  
  return RADNLPModel(meta, counters, f, x->T[], ∇f!, cfg, cfH, nothing, nothing)
end

function RADNLPModel(f, 
                     x0        :: AbstractVector{T},
                     c,
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
  nnzh, cfℓ = _meta_function(f, c, x0, ncon, nvar, Val(hess_free))

  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, ncon=ncon, y0=y0, lcon=lcon, ucon=ucon,
                      nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true,
                      islp=false, name=name)

  counters = Counters()

  return RADNLPModel(meta, counters, f, c, ∇f!, cfg, cfH, cfJ, cfℓ)
end

function RADNLPModel(f, 
                     x0        :: AbstractVector{T},
                     lvar      :: AbstractVector, 
                     uvar      :: AbstractVector,
                     c,
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
  nnzh, cfℓ = _meta_function(f, c, x0, ncon, nvar, Val(hess_free))

  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar,
    ncon=ncon, y0=y0, lcon=lcon, ucon=ucon,
    nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true,
    islp=false, name=name)

  counters = Counters()

  return RADNLPModel(meta, counters, f, c, ∇f!, cfg, cfH, cfJ, cfℓ)
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
  I, J, _ = findnz(Jx)
  rows .= I
  cols .= J
  return rows, cols
end

function NLPModels.jac_coord!(model :: RADNLPModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck model.meta.nvar x
  @lencheck model.meta.nnzj vals
  increment!(model, :neval_jac)
  _fun  = eval(model.cfJ[2])
  Base.invokelatest(_fun, vals, x)
  return vals
end

function NLPModels.jac(model :: RADNLPModel, x :: AbstractVector)
  @lencheck model.meta.nvar x
  increment!(model, :neval_jac)

  if isnothing(model.cfJ) throw(error("This is a matrix-free ADNLPModel.")) end
  _fun = eval(model.cfJ[1])
  rows, cols = jac_structure(model)
  vals = Base.invokelatest(_fun, x)
  return sparse(rows, cols, vals, model.meta.ncon, model.meta.nvar)
end

function NLPModels.jprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck model.meta.nvar x v
  @lencheck model.meta.ncon Jv
  increment!(model, :neval_jprod)
  #Option 1: ForwardDiff
  Jv .= ForwardDiff.derivative(t -> model.c(x + t * v), 0)
  #Option 2: SparseDiffTools
  #Jv .= auto_jacvec(f, x, v)
  return Jv
end

function NLPModels.jtprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck model.meta.nvar x Jtv
  @lencheck model.meta.ncon v
  increment!(model, :neval_jtprod)
  #Option 1: ForwardDiff
  Jtv .= ForwardDiff.gradient(x -> dot(model.c(x), v), x) #ReverseDiff without preparation isn't better
  return Jtv
end

function NLPModels.hess_structure!(model :: RADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck model.meta.nnzh rows cols
  if model.meta.ncon > 0
    @variables xs[1:model.meta.nvar]
    @variables ys[1:model.meta.ncon]
    _fun = model.f(xs) + dot(ys, model.c(xs))
  else
    @variables xs[1:model.meta.nvar]
    _fun = model.f(xs)
  end
  H = tril(Symbolics.hessian_sparsity(_fun, xs))
  I, J, _ = findnz(H)
  rows .= I
  cols .= J
  return rows, cols
end

function NLPModels.hess_coord!(model :: RADNLPModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x
  @lencheck model.meta.nnzh vals
  increment!(model, :neval_hess)
  #Tangi: the only solution I found to get the right pattern...
  if model.meta.ncon > 0
    if isnothing(model.cfℓ) throw(error("This is a matrix-free ADNLPModel.")) end
    _fun = eval(model.cfℓ[2])
    Base.invokelatest(_fun, vals, x, zeros(model.meta.ncon), obj_weight)
  else
    if obj_weight == 0
      vals .= 0
      return vals
    end 
    if isnothing(model.cfH) throw(error("This is a matrix-free ADNLPModel.")) end
    _fun = eval(model.cfH[2])
    Base.invokelatest(_fun, vals, x)
    vals .*= obj_weight
  end
  return vals
end

function NLPModels.hess_coord!(model :: RADNLPModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x
  @lencheck model.meta.ncon y
  @lencheck model.meta.nnzh vals
  increment!(model, :neval_hess)
  if model.meta.nnzh == 0
    return vals
  end
  if isnothing(model.cfℓ) throw(error("This is a matrix-free ADNLPModel.")) end
  _fun = eval(model.cfℓ[2])
  Base.invokelatest(_fun, vals, x, y, obj_weight)
  return vals
end

function NLPModels.hprod!(model :: RADNLPModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x v Hv
  @lencheck model.meta.ncon y
  increment!(model, :neval_hprod)
  #Option 1: ForwardDiff
  ℓ(x) = obj_weight * model.f(x) + dot(model.c(x), y)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function NLPModels.hprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x v Hv
  increment!(model, :neval_hprod)
  #Option 1
  ℓ(x) = obj_weight * model.f(x)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
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
  elseif model.cfH[3] == 0 || obj_weight == 0
    rows, cols = hess_structure(model)
    return sparse(rows, cols, zeros(T, model.meta.nnzh), model.meta.nvar, model.meta.nvar)
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
  vals = obj_weight * Base.invokelatest(_fun, x)
  rows, cols = hess_structure(model)
  return sparse(rows, cols, vals, model.meta.nvar, model.meta.nvar)
end

function hess(model :: RADNLPModel, x :: AbstractVector{T}, y :: AbstractVector{T}; obj_weight :: Real = one(eltype(x))) where T
  @lencheck model.meta.nvar x
  @lencheck model.meta.ncon y
  increment!(model, :neval_hess)
  if model.meta.nnzh == 0
    return spzeros(T, model.meta.nvar, model.meta.nvar)
  end
  if isnothing(model.cfℓ) throw(error("This is a matrix-free ADNLPModel.")) end
  _fun = eval(model.cfℓ[1])
  vals = Base.invokelatest(_fun, x, y, obj_weight)
  rows, cols = hess_structure(model)
  return sparse(rows, cols, vals, model.meta.nvar, model.meta.nvar) #precompiled function return the lower triangular
end
