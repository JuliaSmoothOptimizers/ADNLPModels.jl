export ADNLSModel

mutable struct ADNLSModel{T, S} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
  adbackend::ADBackend

  # Function
  F
  c
end

ADNLPModels.show_header(io::IO, nls::ADNLSModel) = println(
  io,
  "ADNLSModel - Nonlinear least-squares model with automatic differentiation backend $(nls.adbackend)",
)

"""
    ADNLSModel(F, x0, nequ)
    ADNLSModel(F, x0, nequ, lvar, uvar)
    ADNLSModel(F, x0, nequ, c, lcon, ucon)
    ADNLSModel(F, x0, nequ, lvar, uvar, c, lcon, ucon)

ADNLSModel is an Nonlinear Least Squares model using ForwardDiff to
compute the derivatives.
The problem is defined as

     min  ½‖F(x)‖²
    s.to  lcon ≤ c(x) ≤ ucon
          lvar ≤   x  ≤ uvar

The following keyword arguments are available to all constructors:

- `linequ`: An array of indexes of the linear equations (default: `Int[]`)
- `name`: The name of the model (default: "Generic")

The following keyword arguments are available to the constructors for constrained problems:

- `lin`: An array of indexes of the linear constraints (default: `Int[]`)
- `y0`: An inital estimate to the Lagrangian multipliers (default: zeros)
"""
function ADNLSModel(
  F,
  x0::S,
  nequ::Integer;
  linequ::AbstractVector{<:Integer} = Int[],
  name::String = "Generic",
  adbackend = ForwardDiffAD(length(x0), x -> sum(F(x).^2), x0),
) where {S}
  T = eltype(S)
  nvar = length(x0)

  meta = NLPModelMeta{T, S}(nvar, x0 = x0, name = name)
  nlnequ = setdiff(1:nequ, linequ)
  nls_meta = NLSMeta{T, S}(
    nequ,
    nvar,
    nnzj = nequ * nvar,
    nnzh = div(nvar * (nvar + 1), 2),
    lin = linequ,
    nln = nlnequ,
  )

  return ADNLSModel(meta, nls_meta, NLSCounters(), adbackend, F, x -> T[])
end

function ADNLSModel(
  F,
  x0::S,
  nequ::Integer,
  lvar::S,
  uvar::S;
  linequ::AbstractVector{<:Integer} = Int[],
  name::String = "Generic",
  adbackend = ForwardDiffAD(length(x0), x -> sum(F(x).^2), x0),
) where {S}
  T = eltype(S)
  nvar = length(x0)
  @lencheck nvar lvar uvar

  meta = NLPModelMeta{T, S}(nvar, x0 = x0, lvar = lvar, uvar = uvar, name = name)
  nlnequ = setdiff(1:nequ, linequ)
  nls_meta = NLSMeta{T, S}(
    nequ,
    nvar,
    nnzj = nequ * nvar,
    nnzh = div(nvar * (nvar + 1), 2),
    lin = linequ,
    nln = nlnequ,
  )

  return ADNLSModel(meta, nls_meta, NLSCounters(), adbackend, F, x -> T[])
end

function ADNLSModel(
  F,
  x0::S,
  nequ::Integer,
  c,
  lcon::S,
  ucon::S;
  y0::S = fill!(similar(lcon), zero(eltype(S))),
  lin::AbstractVector{<:Integer} = Int[],
  linequ::AbstractVector{<:Integer} = Int[],
  name::String = "Generic",
  adbackend = ForwardDiffAD(length(x0), length(lcon), x -> sum(F(x).^2), x0),
) where {S}
  T = eltype(S)
  nvar = length(x0)
  ncon = length(lcon)
  @lencheck ncon ucon y0
  nnzj = nvar * ncon

  nln = setdiff(1:ncon, lin)
  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    name = name,
    lin = lin,
    nln = nln,
  )
  nlnequ = setdiff(1:nequ, linequ)
  nls_meta = NLSMeta{T, S}(
    nequ,
    nvar,
    nnzj = nequ * nvar,
    nnzh = div(nvar * (nvar + 1), 2),
    lin = linequ,
    nln = nlnequ,
  )

  return ADNLSModel(meta, nls_meta, NLSCounters(), adbackend, F, c)
end

function ADNLSModel(
  F,
  x0::S,
  nequ::Integer,
  lvar::S,
  uvar::S,
  c,
  lcon::S,
  ucon::S;
  y0::S = fill!(similar(lcon), zero(eltype(S))),
  lin::AbstractVector{<:Integer} = Int[],
  linequ::AbstractVector{<:Integer} = Int[],
  name::String = "Generic",
  adbackend = ForwardDiffAD(length(x0), length(lcon), x -> sum(F(x).^2), x0),
) where {S}
  T = eltype(S)
  nvar = length(x0)
  ncon = length(lcon)
  @lencheck nvar lvar uvar
  @lencheck ncon ucon y0
  nnzj = nvar * ncon

  nln = setdiff(1:ncon, lin)
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
    name = name,
    lin = lin,
    nln = nln,
  )
  nlnequ = setdiff(1:nequ, linequ)
  nls_meta = NLSMeta{T, S}(
    nequ,
    nvar,
    nnzj = nequ * nvar,
    nnzh = div(nvar * (nvar + 1), 2),
    lin = linequ,
    nln = nlnequ,
  )

  return ADNLSModel(meta, nls_meta, NLSCounters(), adbackend, F, c)
end

function NLPModels.residual!(nls::ADNLSModel, x::AbstractVector, Fx::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  Fx .= nls.F(x)
  return Fx
end

function NLPModels.jac_residual(nls::ADNLSModel, x::AbstractVector)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jac_residual)
  return jacobian(nls.adbackend, nls.F, x)
end

function NLPModels.jac_structure_residual!(
  nls::ADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nls.nls_meta.nnzj rows cols
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  I = ((i, j) for i = 1:m, j = 1:n)
  rows .= getindex.(I, 1)[:]
  cols .= getindex.(I, 2)[:]
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls::ADNLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nnzj vals
  increment!(nls, :neval_jac_residual)
  Jx = jacobian(nls.adbackend, nls.F, x)
  vals .= Jx[:]
  return vals
end

function NLPModels.jprod_residual!(
  nls::ADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nls.meta.nvar x v
  @lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  Jv .= Jprod(nls.adbackend, nls.F, x, v)
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::ADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_jtprod_residual)
  Jtv .= Jtprod(nls.adbackend, nls.F, x, v)
  return Jtv
end

function NLPModels.hess_residual(nls::ADNLSModel, x::AbstractVector, v::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_hess_residual)
  ϕ(x) = dot(nls.F(x), v)
  return Symmetric(hessian(nls.adbackend, ϕ, x), :L)
end

function NLPModels.hess_structure_residual!(
  nls::ADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nls.nls_meta.nnzh rows cols
  n = nls.meta.nvar
  I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function NLPModels.hess_coord_residual!(
  nls::ADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  @lencheck nls.nls_meta.nnzh vals
  increment!(nls, :neval_hess_residual)
  Hx = hessian(nls.adbackend, x -> dot(nls.F(x), v), x)
  k = 1
  for j = 1:(nls.meta.nvar)
    for i = j:(nls.meta.nvar)
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function NLPModels.jth_hess_residual(nls::ADNLSModel, x::AbstractVector, i::Int)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jhess_residual)
  return Symmetric(hessian(nls.adbackend, x -> nls.F(x)[i], x), :L)
end

function NLPModels.hprod_residual!(
  nls::ADNLSModel,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  @lencheck nls.meta.nvar x v Hiv
  increment!(nls, :neval_hprod_residual)
  Hiv .= Hvprod(nls.adbackend, x -> nls.F(x)[i], x, v)
  return Hiv
end

function NLPModels.cons!(nls::ADNLSModel, x::AbstractVector, c::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.ncon c
  increment!(nls, :neval_cons)
  c .= nls.c(x)
  return c
end

function NLPModels.jac(nls::ADNLSModel, x::AbstractVector)
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_jac)
  return jacobian(nls.adbackend, nls.c, x)
end

function NLPModels.jac_structure!(
  nls::ADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nls.meta.nnzj rows cols
  return jac_structure!(nls.adbackend, nls, rows, cols)
end

function NLPModels.jac_coord!(nls::ADNLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nnzj vals
  return jac_coord!(nls.adbackend, nls, x, vals)
end

function NLPModels.jprod!(nls::ADNLSModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nls.meta.nvar x v
  @lencheck nls.meta.ncon Jv
  increment!(nls, :neval_jprod)
  Jv .= Jprod(nls.adbackend, nls.c, x, v)
  return Jv
end

function NLPModels.jtprod!(
  nls::ADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.meta.ncon v
  increment!(nls, :neval_jtprod)
  Jtv .= Jtprod(nls.adbackend, nls.c, x, v)
  return Jtv
end

function NLPModels.hess(nls::ADNLSModel, x::AbstractVector; obj_weight::Real = one(eltype(x)))
  @lencheck nls.meta.nvar x
  increment!(nls, :neval_hess)
  ℓ(x) = obj_weight * sum(nls.F(x) .^ 2) / 2
  Hx = hessian(nls.adbackend, ℓ, x)
  return Symmetric(Hx, :L)
end

function NLPModels.hess(
  nls::ADNLSModel,
  x::AbstractVector,
  y::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.ncon y
  increment!(nls, :neval_hess)
  ℓ(x) = obj_weight * sum(nls.F(x) .^ 2) / 2 + dot(y, nls.c(x))
  Hx = hessian(nls.adbackend, ℓ, x)
  return Symmetric(Hx, :L)
end

function NLPModels.hess_structure!(
  nls::ADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nls.meta.nnzh rows cols
  return hess_structure!(nls.adbackend, nls, rows, cols)
end

function NLPModels.hess_coord!(
  nls::ADNLSModel,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nnzh vals
  increment!(nls, :neval_hess)
  ℓ(x) = obj_weight * sum(nls.F(x) .^ 2) / 2
  return hess_coord!(nls.adbackend, nls, x, ℓ, vals)
end

function NLPModels.hess_coord!(
  nls::ADNLSModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.ncon y
  @lencheck nls.meta.nnzh vals
  increment!(nls, :neval_hess)
  ℓ(x) = obj_weight * sum(nls.F(x) .^ 2) / 2 + dot(y, nls.c(x))
  return hess_coord!(nls.adbackend, nls, x, ℓ, vals)
end

function NLPModels.hprod!(
  nls::ADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(eltype(x)),
)
  @lencheck nls.meta.nvar x v Hv
  increment!(nls, :neval_hprod)
  ℓ(x) = obj_weight * sum(nls.F(x) .^ 2) / 2
  Hv .= Hvprod(nls.adbackend, ℓ, x, v)
  return Hv
end

function NLPModels.hprod!(
  nls::ADNLSModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(eltype(x)),
)
  @lencheck nls.meta.nvar x v Hv
  @lencheck nls.meta.ncon y
  increment!(nls, :neval_hprod)
  ℓ(x) = obj_weight * sum(nls.F(x) .^ 2) / 2 + dot(y, nls.c(x))
  Hv .= Hvprod(nls.adbackend, ℓ, x, v)
  return Hv
end

function NLPModels.jth_hess_coord!(
  nls::ADNLSModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector,
)
  @lencheck nls.meta.nnzh vals
  @lencheck nls.meta.nvar x
  @rangecheck 1 nls.meta.ncon j
  increment!(nls, :neval_jhess)
  return hess_coord!(nls.adbackend, nls, x, x -> nls.c(x)[j], vals)
end

function NLPModels.jth_hprod!(
  nls::ADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
)
  @lencheck nls.meta.nvar x v Hv
  @rangecheck 1 nls.meta.ncon j
  increment!(nls, :neval_jhprod)
  Hv .= Hvprod(nls.adbackend, x -> nls.c(x)[j], x, v)
  return Hv
end

function NLPModels.ghjvprod!(
  nls::ADNLSModel,
  x::AbstractVector,
  g::AbstractVector,
  v::AbstractVector,
  gHv::AbstractVector,
)
  @lencheck nls.meta.nvar x g v
  @lencheck nls.meta.ncon gHv
  increment!(nls, :neval_hprod)
  gHv .= directional_second_derivative(nls.adbackend, nls.c, x, v, g)
  return gHv
end
