module ADNLPModelsEnzymeExt

using Enzyme, ADNLPModels

function _gradient!(dx, f, x)
  Enzyme.make_zero!(dx)
  Enzyme.autodiff(
    Enzyme.set_runtime_activity(Enzyme.Reverse),
    f,
    Enzyme.Active,
    Enzyme.Duplicated(x, dx),
  )
  return nothing
end

function _hvp!(res, f, x, v)
  Enzyme.autodiff(
    Enzyme.set_runtime_activity(Enzyme.Forward),
    _gradient!,
    res,
    Enzyme.Const(f),
    Enzyme.Duplicated(x, v),
  )
  return nothing
end

function _gradient!(dx, ℓ, x, y, obj_weight, cx)
  Enzyme.make_zero!(dx)
  dcx = Enzyme.make_zero(cx)
  Enzyme.autodiff(
    Enzyme.set_runtime_activity(Enzyme.Reverse),
    ℓ,
    Enzyme.Active,
    Enzyme.Duplicated(x, dx),
    Enzyme.Const(y),
    Enzyme.Const(obj_weight),
    Enzyme.Duplicated(cx, dcx),
  )
  return nothing
end

function _hvp!(res, ℓ, x, v, y, obj_weight, cx)
  dcx = Enzyme.make_zero(cx)
  Enzyme.autodiff(
    Enzyme.set_runtime_activity(Enzyme.Forward),
    _gradient!,
    res,
    Enzyme.Const(ℓ),
    Enzyme.Duplicated(x, v),
    Enzyme.Const(y),
    Enzyme.Const(obj_weight),
    Enzyme.Duplicated(cx, dcx),
  )
  return nothing
end

function ADNLPModels.gradient(::ADNLPModels.EnzymeReverseADGradient, f, x)
  g = similar(x)
  Enzyme.autodiff(
    Enzyme.set_runtime_activity(Enzyme.Reverse),
    Enzyme.Const(f),
    Enzyme.Active,
    Enzyme.Duplicated(x, g),
  )
  return g
end

function ADNLPModels.gradient!(::ADNLPModels.EnzymeReverseADGradient, g, f, x)
  Enzyme.make_zero!(g)
  Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(f), Enzyme.Active, Enzyme.Duplicated(x, g))
  return g
end

jacobian(::ADNLPModels.EnzymeReverseADJacobian, f, x) = Enzyme.jacobian(Enzyme.Reverse, f, x)

function hessian(b::ADNLPModels.EnzymeReverseADHessian, f, x)
  T = eltype(x)
  n = length(x)
  hess = zeros(T, n, n)
  fill!(b.seed, zero(T))
  for i = 1:n
    b.seed[i] = one(T)
    grad = Enzyme.make_zero(x)
    _hvp!(Enzyme.DuplicatedNoNeed(grad, b.Hv), f, x, b.seed)
    view(hess, :, i) .= b.Hv
    b.seed[i] = zero(T)
  end
  return hess
end

function Jprod!(b::ADNLPModels.EnzymeReverseADJprod, Jv, c!, x, v, ::Val)
  copyto!(b.xbuf, x)
  copyto!(b.vbuf, v)
  Enzyme.autodiff(
    Enzyme.Forward,
    Enzyme.Const(c!),
    Enzyme.Duplicated(b.cx, b.jvbuf),
    Enzyme.Duplicated(b.xbuf, b.vbuf),
  )
  copyto!(Jv, b.jvbuf)
  return Jv
end

# Wrapper that calls c!(y, x) but returns nothing.
# Enzyme reverse mode requires functions to return nothing (not their output array),
# otherwise it errors with "Duplicated Returns not yet handled".
function _void_c!(c!, y, x)
  c!(y, x)
  return nothing
end

function Jtprod!(b::ADNLPModels.EnzymeReverseADJtprod, Jtv, c!, x, v, ::Val)
  copyto!(b.xbuf, x)
  copyto!(b.vbuf, v)
  Enzyme.make_zero!(b.jtvbuf)
  Enzyme.autodiff(
    Enzyme.Reverse,
    Enzyme.Const(_void_c!),
    Enzyme.Const(c!),
    Enzyme.Duplicated(b.cx, b.vbuf),
    Enzyme.Duplicated(b.xbuf, b.jtvbuf),
  )
  copyto!(Jtv, b.jtvbuf)
  return Jtv
end

function Hvprod!(
  b::ADNLPModels.EnzymeReverseADHvprod,
  Hv,
  x,
  v,
  ℓ_unused,
  ::Val{:lag},
  y,
  obj_weight::Real = one(eltype(x)),
)
  copyto!(b.xbuf, x)
  copyto!(b.vbuf, v)
  _hvp!(Enzyme.DuplicatedNoNeed(b.grad, b.hvbuf), b.ℓ, b.xbuf, b.vbuf, y, obj_weight, b.cx)
  copyto!(Hv, b.hvbuf)
  return Hv
end

function Hvprod!(
  b::ADNLPModels.EnzymeReverseADHvprod,
  Hv,
  x,
  v,
  f_unused,
  ::Val{:obj},
  obj_weight::Real = one(eltype(x)),
)
  copyto!(b.xbuf, x)
  copyto!(b.vbuf, v)
  _hvp!(Enzyme.DuplicatedNoNeed(b.grad, b.hvbuf), b.f, b.xbuf, b.vbuf)
  @. Hv = obj_weight * b.hvbuf
  return Hv
end

# jth_hprod: Hessian-vector product for the j-th constraint.
# Uses the Lagrangian with y = e_j (unit vector) and obj_weight = 0,
# avoiding the closure x -> c(x)[j] that Enzyme can't handle.
function NLPModels.hprod!(
  b::ADNLPModels.EnzymeReverseADHvprod,
  nlp::ADModel,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
)
  copyto!(b.xbuf, x)
  copyto!(b.vbuf, v)
  b.cx .= 0
  # Build y = e_{j-nlin} (unit vector for nonlinear constraint index)
  b.ybuf .= 0
  k = 0
  for i in nlp.meta.nln
    k += 1
    if i == j
      b.ybuf[k] = one(eltype(x))
      break
    end
  end
  _hvp!(
    Enzyme.DuplicatedNoNeed(b.grad, b.hvbuf),
    b.ℓ,
    b.xbuf,
    b.vbuf,
    b.ybuf,
    zero(eltype(x)),
    b.cx,
  )
  copyto!(Hv, b.hvbuf)
  return Hv
end

# hprod_residual: Hessian-vector product for the i-th residual.
# Uses forward-over-reverse on F_i(x) = F(x)[i].
function NLPModels.hprod_residual!(
  b::ADNLPModels.EnzymeReverseADHvprod,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  i::Integer,
  Hv::AbstractVector,
)
  F = get_F(nls)  # out-of-place version
  Fi(x) = F(x)[i]
  copyto!(b.xbuf, x)
  copyto!(b.vbuf, v)
  _hvp!(Enzyme.DuplicatedNoNeed(b.grad, b.hvbuf), Fi, b.xbuf, b.vbuf)
  copyto!(Hv, b.hvbuf)
  return Hv
end

# Sparse Jacobian
function get_nln_nnzj(b::ADNLPModels.SparseEnzymeADJacobian, nvar, ncon)
  length(b.rowval)
end

function NLPModels.jac_structure!(
  b::ADNLPModels.SparseEnzymeADJacobian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rowval
  for i = 1:(nlp.meta.nvar)
    for j = b.colptr[i]:(b.colptr[i + 1] - 1)
      cols[j] = i
    end
  end
  return rows, cols
end

function sparse_jac_coord!(
  c!::Function,
  b::ADNLPModels.SparseEnzymeADJacobian,
  x::AbstractVector,
  vals::AbstractVector,
)
  # SparseMatrixColorings.jl requires a SparseMatrixCSC for the decompression
  A = SparseMatrixCSC(b.ncon, b.nvar, b.colptr, b.rowval, b.nzval)

  # Enzyme.Duplicated requires primal and shadow to have the same type.
  # Copy x into a pre-allocated buffer to ensure type match with b.v.
  copyto!(b.xbuf, x)

  groups = column_groups(b.result_coloring)
  for (icol, cols) in enumerate(groups)
    # Update the seed
    b.v .= 0
    for col in cols
      b.v[col] = 1
    end

    # b.compressed_jacobian is just a vector Jv here
    # We don't use the vector mode
    Enzyme.autodiff(
      Enzyme.Forward,
      Enzyme.Const(c!),
      Enzyme.Duplicated(b.cx, b.compressed_jacobian),
      Enzyme.Duplicated(b.xbuf, b.v),
    )

    # Update the columns of the Jacobian that have the color `icol`
    decompress_single_color!(A, b.compressed_jacobian, icol, b.result_coloring)
  end
  vals .= b.nzval
  return vals
end

function NLPModels.jac_coord!(
  b::ADNLPModels.SparseEnzymeADJacobian,
  nlp::ADModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  sparse_jac_coord!(nlp.c!, b, x, vals)
  return vals
end

function NLPModels.jac_structure_residual!(
  b::ADNLPModels.SparseEnzymeADJacobian,
  nls::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rowval
  for i = 1:(nls.meta.nvar)
    for j = b.colptr[i]:(b.colptr[i + 1] - 1)
      cols[j] = i
    end
  end
  return rows, cols
end

function NLPModels.jac_coord_residual!(
  b::ADNLPModels.SparseEnzymeADJacobian,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  sparse_jac_coord!(nls.F!, b, x, vals)
  return vals
end

# Sparse Hessian
function get_nln_nnzh(b::ADNLPModels.SparseEnzymeADHessian, nvar)
  return length(b.rowval)
end

function NLPModels.hess_structure!(
  b::ADNLPModels.SparseEnzymeADHessian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rowval
  for i = 1:(nlp.meta.nvar)
    for j = b.colptr[i]:(b.colptr[i + 1] - 1)
      cols[j] = i
    end
  end
  return rows, cols
end

function sparse_hess_coord!(
  b::ADNLPModels.SparseEnzymeADHessian,
  x::AbstractVector,
  obj_weight,
  y::AbstractVector,
  vals::AbstractVector,
)
  # SparseMatrixColorings.jl requires a SparseMatrixCSC for the decompression
  A = SparseMatrixCSC(b.nvar, b.nvar, b.colptr, b.rowval, b.nzval)

  # Enzyme.Duplicated requires primal and shadow to have the same type.
  # Copy x into a pre-allocated buffer to ensure type match with b.v.
  copyto!(b.xbuf, x)

  groups = column_groups(b.result_coloring)
  for (icol, cols) in enumerate(groups)
    # Update the seed
    b.v .= 0
    for col in cols
      b.v[col] = 1
    end

    _hvp!(
      Enzyme.DuplicatedNoNeed(b.grad, b.compressed_hessian_icol),
      b.ℓ,
      b.xbuf,
      b.v,
      y,
      obj_weight,
      b.cx,
    )

    if b.coloring_mode == :direct
      # Update the coefficients of the lower triangular part of the Hessian that are related to the color `icol`
      decompress_single_color!(A, b.compressed_hessian_icol, icol, b.result_coloring, :L)
    end
    if b.coloring_mode == :substitution
      view(b.compressed_hessian, :, icol) .= b.compressed_hessian_icol
    end
  end
  if b.coloring_mode == :substitution
    decompress!(A, b.compressed_hessian, b.result_coloring, :L)
  end
  vals .= b.nzval
  return vals
end

function NLPModels.hess_coord!(
  b::ADNLPModels.SparseEnzymeADHessian,
  nlp::ADModel,
  x::AbstractVector,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  sparse_hess_coord!(b, x, obj_weight, y, vals)
end

# Could be optimized!
function NLPModels.hess_coord!(
  b::ADNLPModels.SparseEnzymeADHessian,
  nlp::ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  b.y .= 0
  sparse_hess_coord!(b, x, obj_weight, b.y, vals)
end

function NLPModels.hess_coord!(
  b::ADNLPModels.SparseEnzymeADHessian,
  nlp::ADModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector,
)
  for (w, k) in enumerate(nlp.meta.nln)
    b.y[w] = k == j ? 1 : 0
  end
  obj_weight = zero(eltype(x))
  sparse_hess_coord!(b, x, obj_weight, b.y, vals)
  return vals
end

function NLPModels.hess_structure_residual!(
  b::ADNLPModels.SparseEnzymeADHessian,
  nls::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  return hess_structure!(b, nls, rows, cols)
end

function NLPModels.hess_coord_residual!(
  b::ADNLPModels.SparseEnzymeADHessian,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  obj_weight = zero(eltype(x))
  sparse_hess_coord!(b, x, obj_weight, v, vals)
end

end
