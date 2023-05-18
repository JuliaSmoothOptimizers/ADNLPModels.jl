abstract type ADBackend end

abstract type ImmutableADbackend <: ADBackend end
abstract type InPlaceADbackend <: ADBackend end
struct EmptyADbackend <: ADBackend end

"""
    get_nln_nnzj(::ADBackend, nvar, ncon)
    get_nln_nnzj(b::ADModelBackend, nvar, ncon)

For a given `ADBackend` of a problem with `nvar` variables and `ncon` constraints, return the number of nonzeros in the Jacobian of nonlinear constraints.
If `b` is the `ADModelBackend` then `b.jacobian_backend` is used.
"""
function get_nln_nnzj(b::ADModelBackend, nvar, ncon)
  get_nln_nnzj(b.jacobian_backend, nvar, ncon)
end

function get_nln_nnzj(::ADBackend, nvar, ncon)
  nvar * ncon
end

"""
    get_residual_nnzj(b::ADModelBackend, nvar, nequ)

Return `get_nln_nnzj(b.jacobian_residual_backend, nvar, nequ)`.
"""
function get_residual_nnzj(b::ADModelBackend, nvar, nequ)
  get_nln_nnzj(b.jacobian_residual_backend, nvar, nequ)
end

"""
    get_nln_nnzh(::ADBackend, nvar)
    get_nln_nnzh(b::ADModelBackend, nvar)

For a given `ADBackend` of a problem with `nvar` variables, return the number of nonzeros in the lower triangle of the Hessian.
If `b` is the `ADModelBackend` then `b.hessian_backend` is used.
"""
function get_nln_nnzh(b::ADModelBackend, nvar)
  get_nln_nnzh(b.hessian_backend, nvar)
end

function get_nln_nnzh(::ADBackend, nvar)
  div(nvar * (nvar + 1), 2)
end

throw_error(b) =
  throw(ArgumentError("The AD backend $b is not loaded. Please load the corresponding AD package."))
gradient(b::ADBackend, ::Any, ::Any) = throw_error(b)
gradient!(b::ADBackend, ::Any, ::Any, ::Any) = throw_error(b)
jacobian(b::ADBackend, ::Any, ::Any) = throw_error(b)
hessian(b::ADBackend, ::Any, ::Any) = throw_error(b)
Jprod!(b::ADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)
Jtprod!(b::ADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)
Hvprod!(b::ADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)
directional_second_derivative(::ADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)

function hess_structure!(
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

function hess_coord!(
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

function hess_coord!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ = get_lag(nlp, b, obj_weight)
  return hess_coord!(b, nlp, x, ℓ, vals)
end

function hess_coord!(
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

function hess_coord!(
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

function hprod!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
)
  c = get_c(nlp, b)
  Hvprod!(b, Hv, x -> c(x)[j - nlp.meta.nlin], x, v)
  return Hv
end

function jac_structure!(
  b::ADBackend,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  m, n = nlp.meta.nnln, nlp.meta.nvar
  return jac_dense!(m, n, rows, cols)
end

function jac_structure_residual!(
  b::ADBackend,
  nls::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  return jac_dense!(m, n, rows, cols)
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

function jac_coord!(b::ADBackend, nlp::ADModel, x::AbstractVector, vals::AbstractVector)
  c = get_c(nlp, b)
  Jx = jacobian(b, c, x)
  vals .= view(Jx, :)
  return vals
end

function jac_coord_residual!(
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
