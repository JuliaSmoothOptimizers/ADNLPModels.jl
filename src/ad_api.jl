abstract type ADBackend end

abstract type ImmutableADbackend <: ADBackend end
abstract type InPlaceADbackend <: ADBackend end

struct EmptyADbackend <: ADBackend end
EmptyADbackend(args...; kwargs...) = EmptyADbackend()

function Base.show(
  io::IO,
  backend::ADModelBackend{GB, HvB, JvB, JtvB, JB, HB, GHJ, HvBLS, JvBLS, JtvBLS, JBLS, HBLS},
) where {
  GB,
  HvB,
  JvB,
  JtvB,
  JB,
  HB,
  GHJ,
  HvBLS <: EmptyADbackend,
  JvBLS <: EmptyADbackend,
  JtvBLS <: EmptyADbackend,
  JBLS <: EmptyADbackend,
  HBLS <: EmptyADbackend,
}
  print(io, replace(replace(
    "ADModelBackend{
  $GB,
  $HvB,
  $JvB,
  $JtvB,
  $JB,
  $HB,
  $GHJ,
}",
    "ADNLPModels." => "",
  ), r"\{(.+)\}" => s""))
end

function Base.show(
  io::IO,
  backend::ADModelBackend{GB, HvB, JvB, JtvB, JB, HB, GHJ, HvBLS, JvBLS, JtvBLS, JBLS, HBLS},
) where {GB, HvB, JvB, JtvB, JB, HB, GHJ, HvBLS, JvBLS, JtvBLS, JBLS, HBLS}
  print(io, replace(replace(
    "ADModelBackend{
  $GB,
  $HvB,
  $JvB,
  $JtvB,
  $JB,
  $HB,
  $GHJ,
  $HvBLS,
  $JvBLS,
  $JtvBLS,
  $JBLS,
  $HBLS,
}",
    "ADNLPModels." => "",
  ), r"\{(.+)\}" => s""))
end

"""
    get_nln_nnzj(::ADBackend, nvar, ncon)
    get_nln_nnzj(b::ADModelBackend, nvar, ncon)
    get_nln_nnzj(nlp::AbstractNLPModel, nvar, ncon)

For a given `ADBackend` of a problem with `nvar` variables and `ncon` constraints, return the number of nonzeros in the Jacobian of nonlinear constraints.
If `b` is the `ADModelBackend` then `b.jacobian_backend` is used.
"""
function get_nln_nnzj(b::ADModelBackend, nvar, ncon)
  get_nln_nnzj(b.jacobian_backend, nvar, ncon)
end

function get_nln_nnzj(::ADBackend, nvar, ncon)
  nvar * ncon
end

function get_nln_nnzj(nlp::AbstractNLPModel, nvar, ncon)
  nlp.meta.nln_nnzj
end

"""
    get_residual_nnzj(b::ADModelBackend, nvar, nequ)

Return `get_nln_nnzj(b.jacobian_residual_backend, nvar, nequ)`.
"""
function get_residual_nnzj(b::ADModelBackend, nvar, nequ)
  get_nln_nnzj(b.jacobian_residual_backend, nvar, nequ)
end

function get_residual_nnzj(
  b::ADModelBackend{GB, HvB, JvB, JtvB, JB, HB, GHJ, HvBLS, JvBLS, JtvBLS, JBLS, HBLS},
  nvar,
  nequ,
) where {GB, HvB, JvB, JtvB, JB, HB, GHJ, HvBLS, JvBLS, JtvBLS, JBLS <: AbstractNLPModel, HBLS}
  nls = b.jacobian_residual_backend
  nls.nls_meta.nnzj
end

"""
    get_nln_nnzh(::ADBackend, nvar)
    get_nln_nnzh(b::ADModelBackend, nvar)
    get_nln_nnzh(nlp::AbstractNLPModel, nvar)

For a given `ADBackend` of a problem with `nvar` variables, return the number of nonzeros in the lower triangle of the Hessian.
If `b` is the `ADModelBackend` then `b.hessian_backend` is used.
"""
function get_nln_nnzh(b::ADModelBackend, nvar)
  get_nln_nnzh(b.hessian_backend, nvar)
end

function get_nln_nnzh(::ADBackend, nvar)
  div(nvar * (nvar + 1), 2)
end

function get_nln_nnzh(nlp::AbstractNLPModel, nvar)
  nlp.meta.nnzh
end

throw_error(b) =
  throw(ArgumentError("The AD backend $b is not loaded. Please load the corresponding AD package."))
gradient(b::ADBackend, ::Any, ::Any) = throw_error(b)
gradient!(b::ADBackend, ::Any, ::Any, ::Any) = throw_error(b)
jacobian(b::ADBackend, ::Any, ::Any) = throw_error(b)
hessian(b::ADBackend, ::Any, ::Any) = throw_error(b)
Jprod!(b::ADBackend, ::Any, ::Any, ::Any, ::Any, ::Any) = throw_error(b)
Jtprod!(b::ADBackend, ::Any, ::Any, ::Any, ::Any, ::Any) = throw_error(b)
Hvprod!(b::ADBackend, ::Any, ::Any, ::Any, ::Any, ::Any, args...) = throw_error(b)
directional_second_derivative(::ADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)

# API for AbstractNLPModel as backend
gradient(nlp::AbstractNLPModel, f, x) = grad(nlp, x)
gradient!(nlp::AbstractNLPModel, g, f, x) = grad!(nlp, x, g)
Jprod!(nlp::AbstractNLPModel, Jv, c, x, v, ::Val{:c}) = jprod_nln!(nlp, x, v, Jv)
Jprod!(nlp::AbstractNLPModel, Jv, c, x, v, ::Val{:F}) = jprod_residual!(nlp, x, v, Jv)
Jtprod!(nlp::AbstractNLPModel, Jtv, c, x, v, ::Val{:c}) = jtprod_nln!(nlp, x, v, Jtv)
Jtprod!(nlp::AbstractNLPModel, Jtv, c, x, v, ::Val{:F}) = jtprod_residual!(nlp, x, v, Jtv)
function Hvprod!(nlp::AbstractNLPModel, Hv, x, v, ℓ, ::Val{:obj}, obj_weight)
  return hprod!(nlp, x, v, Hv, obj_weight = obj_weight)
end
function Hvprod!(nlp::AbstractNLPModel, Hv, x::S, v, ℓ, ::Val{:lag}, y, obj_weight) where {S}
  if nlp.meta.nlin > 0
    # y is of length nnln, and hprod expectes ncon...
    yfull = fill!(S(undef, nlp.meta.ncon), 0)
    k = 0
    for i in nlp.meta.nln
      k += 1
      yfull[i] = y[k]
    end
    return hprod!(nlp, x, yfull, v, Hv, obj_weight = obj_weight)
  end
  return hprod!(nlp, x, y, v, Hv, obj_weight = obj_weight)
end
function directional_second_derivative(nlp::AbstractNLPModel, c, x, v, g)
  gHv = ghjvprod(nlp, x, g, v)
  return view(gHv, (nlp.meta.nlin + 1):(nlp.meta.ncon))
end

function NLPModels.hess_structure!(
  b::ADBackend,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  n = nlp.meta.nvar
  pos = 0
  for j = 1:n
    for i = j:n
      pos += 1
      rows[pos] = i
      cols[pos] = j
    end
  end
  return rows, cols
end

function NLPModels.hess_structure!(
  nlp::AbstractNLPModel,
  ::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  return NLPModels.hess_structure!(nlp, rows, cols)
end

function NLPModels.hess_coord!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ = get_lag(nlp, b, obj_weight, y)
  hess_coord!(b, nlp, x, ℓ, vals)
  return vals
end

function NLPModels.hess_coord!(
  nlp::AbstractNLPModel,
  ::ADModel,
  x::S,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
) where {S}
  if nlp.meta.nlin > 0
    # y is of length nnln, and hess expectes ncon...
    yfull = fill!(S(undef, nlp.meta.ncon), 0)
    k = 0
    for i in nlp.meta.nln
      k += 1
      yfull[i] = y[k]
    end
    return hess_coord!(nlp, x, yfull, vals, obj_weight = obj_weight)
  end
  return hess_coord!(nlp, x, y, vals, obj_weight = obj_weight)
end

function NLPModels.hess_coord!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ = get_lag(nlp, b, obj_weight)
  return hess_coord!(b, nlp, x, ℓ, vals)
end

function NLPModels.hess_coord!(
  nlp::AbstractNLPModel,
  ::ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  return NLPModels.hess_coord!(nlp, x, vals, obj_weight = obj_weight)
end

function NLPModels.hess_coord!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector,
)
  c = get_c(nlp, b)
  ℓ = x -> c(x)[j - nlp.meta.nlin]
  Hx = hessian(b, ℓ, x)
  k = 1
  n = nlp.meta.nvar
  for j = 1:n
    for i = j:n
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function NLPModels.hess_coord!(
  nlp::AbstractNLPModel,
  ::ADModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector,
)
  return NLPModels.jth_hess_coord!(nlp, x, j, vals)
end

function NLPModels.hess_coord!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  ℓ::Function,
  vals::AbstractVector,
)
  Hx = hessian(b, ℓ, x)
  k = 1
  n = nlp.meta.nvar
  for j = 1:n
    for i = j:n
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function NLPModels.hess_structure_residuals!(
  b::ADBackend,
  nls::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  nothing
end

function NLPModels.hess_coord_residuals!(
  b::ADBackend,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  nothing
end

function NLPModels.hprod!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
)
  c = get_c(nlp, b)
  Hvprod!(b, Hv, x, v, x -> c(x)[j - nlp.meta.nlin], Val(:ci))
  return Hv
end

function NLPModels.hprod!(
  nlp::AbstractNLPModel,
  ::ADModel,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
)
  return jth_hprod!(nlp, x, v, j, Hv)
end

function NLPModels.hprod_residual!(
  b::ADBackend,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  i::Integer,
  Hv::AbstractVector,
)
  F = get_F(nls, nls.adbackend.hprod_residual_backend)
  Hvprod!(nls.adbackend.hprod_residual_backend, Hv, x, v, x -> F(x)[i], Val(:ci))
  return Hv
end

function NLPModels.hprod_residual!(
  nlp::AbstractNLPModel,
  ::AbstractADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  i::Integer,
  Hiv::AbstractVector,
)
  return hprod_residual!(nlp, x, i, v, Hiv)
end

function NLPModels.jac_structure!(
  b::ADBackend,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  m, n = nlp.meta.nnln, nlp.meta.nvar
  return jac_dense!(m, n, rows, cols)
end

function NLPModels.jac_structure!(
  nlp::AbstractNLPModel,
  ::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  return jac_nln_structure!(nlp, rows, cols)
end

function NLPModels.jac_structure_residual!(
  b::ADBackend,
  nls::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  return jac_dense!(m, n, rows, cols)
end

function NLPModels.jac_structure_residual!(
  nlp::AbstractNLPModel,
  ::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  return jac_structure_residual!(nlp, rows, cols)
end

function jac_dense!(
  m::Integer,
  n::Integer,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  pos = 0
  for j = 1:n
    for i = 1:m
      pos += 1
      rows[pos] = i
      cols[pos] = j
    end
  end
  return rows, cols
end

function NLPModels.jac_coord!(b::ADBackend, nlp::ADModel, x::AbstractVector, vals::AbstractVector)
  c = get_c(nlp, b)
  Jx = jacobian(b, c, x)
  vals .= view(Jx, :)
  return vals
end

function NLPModels.jac_coord!(
  nlp::AbstractNLPModel,
  ::ADModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  return jac_nln_coord!(nlp, x, vals)
end

function NLPModels.jac_coord_residual!(
  b::ADBackend,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  F = get_F(nls, b)
  Jx = jacobian(b, F, x)
  vals .= view(Jx, :)
  return vals
end

function NLPModels.jac_coord_residual!(
  nlp::AbstractNLPModel,
  ::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  return jac_coord_residual!(nlp, x, vals)
end
