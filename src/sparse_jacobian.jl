## ----- SparseDiffTools -----

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

function jac_structure_residual!(
  b::SparseForwardADJacobian,
  nls::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= rowvals(b.cfJ.sparsity)
  for i = 1:(nls.meta.nvar)
    for j = b.cfJ.sparsity.colptr[i]:(b.cfJ.sparsity.colptr[i + 1] - 1)
      cols[j] = i
    end
  end
  return rows, cols
end

function jac_coord_residual!(
  b::SparseForwardADJacobian,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  forwarddiff_color_jacobian!(b.cfJ.sparsity, nls.F!, x, b.cfJ)
  vals .= nonzeros(b.cfJ.sparsity)
  return vals
end

## ----- Symbolics -----

struct SparseADJacobian <: ADBackend
  nnzj::Int
  rows::Vector{Int}
  cols::Vector{Int}
  cfJ::Expr
end

function SparseADJacobian(nvar, f, ncon, c!; kwargs...)
  @variables xs[1:nvar] out[1:ncon]
  wi = Symbolics.scalarize(xs)
  wo = Symbolics.scalarize(out)
  fun = c!(wo, wi)
  S = Symbolics.jacobian_sparsity(c!, wo, wi)
  rows, cols, _ = findnz(S)
  vals = Symbolics.sparsejacobian_vals(fun, wi, rows, cols)
  nnzj = length(vals)
  # cfJ is a Tuple{Expr, Expr}, cfJ[2] is the in-place function
  # that we need to update a vector `vals` with the nonzeros of Jc(x).
  cfJ = Symbolics.build_function(vals, wi)
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

function jac_structure_residual!(
  b::SparseADJacobian,
  nls::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end

function jac_coord_residual!(
  b::SparseADJacobian,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  @eval $(b.cfJ)($vals, $x)
  return vals
end
