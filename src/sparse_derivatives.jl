## ----- Symbolics -----

# Jacobian
struct SparseADJacobian{J} <: ADBackend
  nnzj::Int
  rows::Vector{Int}
  cols::Vector{Int}
  cfJ::J
end

function SparseADJacobian(nvar, f, ncon, c!; kwargs...)
  @variables xs[1:nvar] out[1:ncon]
  wi = Symbolics.scalarize(xs)
  wo = Symbolics.scalarize(out)
  _fun = c!(wo, wi)
  S = Symbolics.jacobian_sparsity(_fun, wi)
  rows, cols, _ = findnz(S)
  vals = Symbolics.sparsejacobian_vals(_fun, wi, rows, cols)
  nnzj = length(vals)
  # cfJ is a Tuple{Expr, Expr}, cfJ[2] is the in-place function
  # that we need to update a vector `vals` with the nonzeros of Jc(x).
  cfJ = Symbolics.build_function(vals, wi, expression = Val{false})
  SparseADJacobian(nnzj, rows, cols, cfJ[2])
end

function get_nln_nnzj(b::SparseADJacobian, nvar, ncon)
  b.nnzj
end

function jac_structure!(
  b::SparseADJacobian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end

function jac_coord!(b::SparseADJacobian, nlp::ADModel, x::AbstractVector, vals::AbstractVector)
  @eval $(b.cfJ)($vals, $x)
  return vals
end

# Hessian
struct SparseADHessian{H} <: ADBackend
  nnzh::Int
  rows::Vector{Int}
  cols::Vector{Int}
  cfH::H
end

function SparseADHessian(nvar, f, ncon, c!; x0::AbstractVector = rand(nvar), kwargs...)
  @variables xs[1:nvar]
  Tv = eltype(x0)
  ℓ(x) = nothing
  if ncon > 0
    cx = rand(Tv, ncon)
    y = rand(Tv, ncon)
    ℓ(x) = f(x) + dot(c!(cx,x), y)
  else
    ℓ(x) = f(x)
  end
  wi = Symbolics.scalarize(xs)
  _fun = ℓ(wi)
  S = Symbolics.hessian_sparsity(_fun, wi, full=false)
  rows, cols, _ = findnz(S)
  vals = Symbolics.sparsehessian_vals(_fun, wi, rows, cols)
  nnzh = length(vals)
  # cfH is a Tuple{Expr, Expr}, cfH[2] is the in-place function
  # that we need to update a vector `vals` with the nonzeros of ∇²f(x).
  cfH = Symbolics.build_function(vals, wi, expression = Val{false})
  SparseADHessian(nnzh, rows, cols, cfH[2])
end

function get_nln_nnzh(b::SparseADHessian, nvar)
  b.nnzh
end

function hess_structure!(
  b::SparseADHessian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end

function hess_coord!(
  b::SparseADHessian,
  nlp::ADModel,
  x::AbstractVector,
  ℓ::Function,
  vals::AbstractVector,
)
  @eval $(b.cfH)($vals, $x)
  return vals
end

## ----- SparseDiffTools -----

# Jacobian
struct SparseForwardADJacobian{Tv, Ti, T, T2, T3, T4, T5} <: ADNLPModels.ADBackend
  cfJ::ForwardColorJacCache{T, T2, T3, T4, T5, SparseMatrixCSC{Tv, Ti}}
end

function SparseForwardADJacobian(nvar, f, ncon, c!;
  x0::AbstractVector = rand(nvar),
  alg::SparseDiffTools.SparseDiffToolsColoringAlgorithm = SparseDiffTools.GreedyD1Color(),
  kwargs...,
)
  Tv = eltype(x0)
  output = similar(x0, ncon)
  sparsity_pattern = Symbolics.jacobian_sparsity(c!, output, x0)
  colors = matrix_colors(sparsity_pattern, alg)
  jac = SparseMatrixCSC{Tv, Int}(
    sparsity_pattern.m,
    sparsity_pattern.n,
    sparsity_pattern.colptr,
    sparsity_pattern.rowval,
    Tv.(sparsity_pattern.nzval),
  )

  dx = zeros(Tv, ncon)
  cfJ = ForwardColorJacCache(c!, x0, colorvec=colors, dx=dx, sparsity=jac)
  SparseForwardADJacobian(cfJ)
end

function get_nln_nnzj(b::SparseForwardADJacobian, nvar, ncon)
  nnz(b.cfJ.sparsity)
end

function jac_structure!(
  b::SparseForwardADJacobian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= rowvals(b.cfJ.sparsity)
  for i = 1:(nlp.meta.nvar)
    for j = b.cfJ.sparsity.colptr[i]:(b.cfJ.sparsity.colptr[i + 1] - 1)
      cols[j] = i
    end
  end
  return rows, cols
end

function jac_coord!(
  b::SparseForwardADJacobian,
  nlp::ADModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  forwarddiff_color_jacobian!(b.cfJ.sparsity, nlp.c!, x, b.cfJ)
  vals .= nonzeros(b.cfJ.sparsity)
  return vals
end

# Hessian
struct SparseForwardADHessian{T, T2, T3, T4} <: ADNLPModels.ADBackend
  cfH::ForwardAutoColorHesCache{T, T2, T3, T4}
end

function SparseForwardADHessian(nvar, f, ncon, c!;
  x0::AbstractVector = rand(nvar),
  alg::SparseDiffTools.SparseDiffToolsColoringAlgorithm = SparseDiffTools.GreedyD1Color(),
  kwargs...,
)
  Tv = eltype(x0)
  ℓ(x) = nothing
  if ncon > 0
    cx = rand(Tv, ncon)
    y = rand(Tv, ncon)
    ℓ(x) = f(x) + dot(c!(cx,x), y)
  else
    ℓ(x) = f(x)
  end
  sparsity_pattern = Symbolics.hessian_sparsity(ℓ, x0, full=false)
  colors = matrix_colors(sparsity_pattern, alg)
  hess = SparseMatrixCSC{Tv, Int}(
    sparsity_pattern.m,
    sparsity_pattern.n,
    sparsity_pattern.colptr,
    sparsity_pattern.rowval,
    Tv.(sparsity_pattern.nzval),
  )

  cfH = ForwardAutoColorHesCache(ℓ, x0, colors, hess)
  SparseForwardADHessian(cfH)
end

function get_nln_nnzh(b::SparseForwardADHessian, nvar)
  nnz(b.cfH.sparsity)
end

function hess_structure!(
  b::SparseForwardADHessian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= rowvals(b.cfH.sparsity)
  for i = 1:(nlp.meta.nvar)
    for j = b.cfH.sparsity.colptr[i]:(b.cfH.sparsity.colptr[i + 1] - 1)
      cols[j] = i
    end
  end
  return rows, cols
end

function hess_coord!(
  b::SparseForwardADHessian,
  nlp::ADModel,
  x::AbstractVector,
  ℓ::Function,
  vals::AbstractVector,
)
  autoauto_color_hessian!(b.cfH.sparsity, ℓ, x, b.cfH)
  vals .= nonzeros(b.cfH.sparsity)
  return vals
end
