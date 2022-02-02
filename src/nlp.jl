export ADNLPModel

mutable struct ADNLPModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  backends::Dict{Tuple{Val,Val}, ADBackend}

  # Functions
  f
  c
end

ADNLPModels.show_header(io::IO, nlp::ADNLPModel) =
  println(io, "ADNLPModel - Model with automatic differentiation")

"""
    ADNLPModel(f, x0)
    ADNLPModel(f, x0, lvar, uvar)
    ADNLPModel(f, x0, c, lcon, ucon)
    ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)

ADNLPModel is an AbstractNLPModel using automatic differentiation to compute the derivatives.
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
function ADNLPModel(f, x0::S; kwargs...) where {S}
  T = eltype(S)
  empty = S(undef, 0)
  nvar = length(x0)
  lvar = fill!(S(undef, nvar), T(-Inf))
  uvar = fill!(S(undef, nvar), T(Inf))
  return ADNLPModel(f, x0, lvar, uvar, nothing, empty, empty; kwargs...)
end

function ADNLPModel(f, x0::S, lvar::S, uvar::S; kwargs...) where {S}
  empty = S(undef, 0)
  return ADNLPModel(f, x0, lvar, uvar, nothing, empty, empty; kwargs...)
end

function ADNLPModel(f, x0::S, c, lcon::S, ucon::S; kwargs...) where {S}
  T = eltype(S)
  nvar = length(x0)
  lvar = fill!(S(undef, nvar), T(-Inf))
  uvar = fill!(S(undef, nvar), T(Inf))
  return ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon; kwargs...)
end

function ADNLPModel(
  f,
  x0::S,
  lvar::S,
  uvar::S,
  _c,
  lcon::S,
  ucon::S;
  y0::S = fill!(similar(lcon), zero(eltype(S))),
  name::String = "Generic",
  lin::AbstractVector{<:Integer} = Int[],
) where {S}
  T = eltype(S)
  nvar = length(x0)
  ncon = length(lcon)
  @lencheck nvar x0
  @lencheck ncon y0 ucon
  if length(lvar) == 0
    @lencheck 0 lvar uvar
  else
    @lencheck nvar lvar uvar
  end
  c = isnothing(_c) ? x -> T[] : _c

  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon

  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    nnzh = nnzh,
    lin = lin,
    minimize = true,
    islp = false,
    name = name,
  )

  backends = Dict{Tuple{Val, Val}, ADBackend}(
    (Val(ForwardDiffAD), Val(T)) => ForwardDiffAD(f, x0),
    (Val(ReverseDiffAD), Val(T)) => ReverseDiffAD(f, x0),
  )

  return ADNLPModel(meta, Counters(), backends, f, c)
end

function get_backend(nlp::ADNLPModel, x::AbstractVector{T}, ::Type{AD}) where {T, AD}
  key = (Val(AD), Val(T))
  if !haskey(nlp.backends, key)
    nlp.backends[key] = AD(nlp.f, x)
  end
  nlp.backends[key]
end

function NLPModels.obj(nlp::ADNLPModel, x::AbstractVector{T}) where {T}
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  return nlp.f(x)::T
end

function NLPModels.grad!(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  g::AbstractVector{T},
  ::Type{AD} = ReverseDiffAD
) where {T, AD <: ADBackend}
  @lencheck nlp.meta.nvar x g
  increment!(nlp, :neval_grad)
  ad = get_backend(nlp, x, AD)
  gradient!(ad, g, nlp.f, x)
  return g
end

function NLPModels.hess(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  ::Type{AD} = ReverseDiffAD;
  obj_weight::T = one(T),
) where {T, AD <: ADBackend}
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x)
  ad = get_backend(nlp, x, AD)
  Hx = hessian(ad, ℓ, x)
  return Symmetric(Hx, :L)
end

function NLPModels.hess(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  ::Type{AD} = ReverseDiffAD;
  obj_weight::T = one(T),
) where {T, AD <: ADBackend}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  ad = get_backend(nlp, x, AD)
  Hx = hessian(ad, ℓ, x)
  return Symmetric(Hx, :L)
end

function NLPModels.hess_structure!(
  nlp::ADNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nnzh rows cols
  n = nlp.meta.nvar
  I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  vals::AbstractVector{T},
  ::Type{AD} = ReverseDiffAD;
  obj_weight::T = one(T),
) where {T, AD <: ADBackend}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x)
  ad = get_backend(nlp, x, AD)
  Hx = hessian(ad, ℓ, x)
  k = 1
  for j = 1:(nlp.meta.nvar)
    for i = j:(nlp.meta.nvar)
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function NLPModels.hess_coord!(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T},
  ::Type{AD} = ReverseDiffAD;
  obj_weight::T = one(T),
) where {T, AD <: ADBackend}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  ad = get_backend(nlp, x, AD)
  Hx = hessian(ad, ℓ, x)
  k = 1
  for j = 1:(nlp.meta.nvar)
    for i = j:(nlp.meta.nvar)
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function NLPModels.cons!(nlp::ADNLPModel, x::AbstractVector{T}, c::AbstractVector) where {T}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)
  c .= nlp.c(x)
  return c
end

function NLPModels.jac(nlp::ADNLPModel, x::AbstractVector{T}, ::Type{AD} = ReverseDiffAD) where {T, AD}
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_jac)
  ad = get_backend(nlp, x, AD)
  return jacobian(ad, nlp.c, x)::Matrix{T}
end

function NLPModels.jac_structure!(
  nlp::ADNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nnzj rows cols
  n = nlp.meta.nvar
  I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function NLPModels.jac_coord!(nlp::ADNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  return jac_coord!(nlp.adbackend, nlp, x, vals)
end

function NLPModels.jprod!(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  v::AbstractVector,
  Jv::AbstractVector,
  ::Type{AD} = ADBackend,
) where {T, AD <: ADBackend}
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)
  ad = get_backend(nlp, x, AD)
  Jv .= Jprod(ad, nlp.c, x, v)
  return Jv
end

function NLPModels.jtprod!(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  v::AbstractVector,
  Jtv::AbstractVector,
  ::Type{AD} = ADBackend,
) where {T, AD <: ADBackend}
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)
  ad= get_backend(nlp, x, AD)
  Jtv .= Jtprod(nlp.adbackend, nlp.c, x, v)
  return Jtv
end

function NLPModels.hprod!(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  v::AbstractVector,
  Hv::AbstractVector,
  ::Type{AD} = ReverseDiffAD;
  obj_weight::T = one(T),
) where {T, AD <: ADBackend}
  n = nlp.meta.nvar
  @lencheck n x v Hv
  increment!(nlp, :neval_hprod)
  ℓ(x) = obj_weight * nlp.f(x)
  ad = get_backend(nlp, x, AD)
  Hv .= Hvprod(ad, ℓ, x, v)
  return Hv
end

function NLPModels.hprod!(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector,
  ::Type{AD} = ReverseDiffAD;
  obj_weight::T = one(T),
) where {T, AD <: ADBackend}
  n = nlp.meta.nvar
  @lencheck n x v Hv
  @lencheck nlp.meta.ncon y
  increment!(nlp, :neval_hprod)
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  ad = get_backend(nlp, x, AD)
  Hv .= Hvprod(ad, ℓ, x, v)
  return Hv
end

function NLPModels.jth_hess_coord!(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  j::Integer,
  vals::AbstractVector,
  ::Type{AD} = ReverseDiffAD,
) where {T, AD <: ADBackend}
  @lencheck nlp.meta.nnzh vals
  @lencheck nlp.meta.nvar x
  @rangecheck 1 nlp.meta.ncon j
  increment!(nlp, :neval_jhess)
  ℓ(x) = nlp.c(x)[j]
  ad = get_backend(nlp, x, AD)
  Hx = hessian(ad, ℓ, x)
  k = 1
  for j = 1:(nlp.meta.nvar)
    for i = j:(nlp.meta.nvar)
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function NLPModels.jth_hprod!(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
  ::Type{AD} = ReverseDiffAD,
) where {T, AD <: ADBackend}
  @lencheck nlp.meta.nvar x v Hv
  @rangecheck 1 nlp.meta.ncon j
  increment!(nlp, :neval_jhprod)
  ad = get_backend(nlp, x, AD)
  Hv .= Hvprod(ad, x -> nlp.c(x)[j], x, v)
  return Hv
end

function NLPModels.ghjvprod!(
  nlp::ADNLPModel,
  x::AbstractVector{T},
  g::AbstractVector,
  v::AbstractVector,
  gHv::AbstractVector,
  ::Type{AD} = ReverseDiffAD,
) where {T, AD <: ADBackend}
  @lencheck nlp.meta.nvar x g v
  @lencheck nlp.meta.ncon gHv
  increment!(nlp, :neval_hprod)
  gHv .= directional_second_derivative(ad, nlp.c, x, v, g)
  return gHv
end
