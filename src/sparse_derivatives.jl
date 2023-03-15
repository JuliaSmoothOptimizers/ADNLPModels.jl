struct SparseForwardADJacobian{Tv, Ti, T, T2, T3, T4, T5} <: ADNLPModels.ADBackend
  cfJ::ForwardColorJacCache{T, T2, T3, T4, T5, SparseMatrixCSC{Tv, Ti}}
end

function SparseForwardADJacobian(nvar, f, ncon, c!; x0::AbstractVector = rand(nvar), alg::SparseDiffTools.SparseDiffToolsColoringAlgorithm = SparseDiffTools.GreedyD1Color(), kwargs...)
  Tv = eltype(x0)
  output = similar(x0, ncon)
  sparsity_pattern = Symbolics.jacobian_sparsity(c!, output, x0)
  colors = matrix_colors(sparsity_pattern, alg)
  jac = SparseMatrixCSC{Tv, Int}(sparsity_pattern.m, sparsity_pattern.n, sparsity_pattern.colptr, sparsity_pattern.rowval, similar(Tv, sparsity_pattern.nzval)) 
  dx = zeros(Tv, ncon)
  sparsity_pattern = Tv.(sparsity_pattern)
  cfJ = ForwardColorJacCache(c!, x0, colorvec = colors, dx = dx, sparsity = jac) 

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
  S = Symbolics.jacobian_sparsity(c!, wo, wi)
  rows, cols, _ = findnz(S)
  vals = Symbolics.sparsejacobian_vals(_fun, wi, rows, cols)
  nnzj = length(rows)
  cfJ = Symbolics.build_function(vals, wi, expression = Val{false})
  SparseADJacobian(nnzj, rows, cols, cfJ)
end

function get_nln_nnzj(b::SparseADJacobian, nvar, ncon)
  b.nnzj
end

function jac_structure!(
  b::SparseADJacobian,
  ::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end
function jac_coord!(b::SparseADJacobian, ::ADModel, x::AbstractVector, vals::AbstractVector)
  _fun = eval(b.cfJ[2])
  Base.invokelatest(_fun, vals, x)
  return vals
end
