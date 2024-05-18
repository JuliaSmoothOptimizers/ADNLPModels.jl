module ADNLPModelsSparseDiffToolsExt

using SparseDiffTools, SparseArrays, ADNLPModels, NLPModels, ForwardDiff
import ADNLPModels: ADModel, AbstractADNLSModel, InPlaceADbackend, ADBackend

function sparse_matrix_colors(A, alg::SparseDiffTools.SparseDiffToolsColoringAlgorithm)
  return SparseDiffTools.matrix_colors(A, alg)
end

struct SDTSparseADJacobian{Tv, Ti, T, T2, T3, T4, T5} <: ADNLPModels.ADBackend
  cfJ::SparseDiffTools.ForwardColorJacCache{T, T2, T3, T4, T5, SparseMatrixCSC{Tv, Ti}}
end

function SDTSparseADJacobian(
  nvar,
  f,
  ncon,
  c!;
  x0::S = rand(nvar),
  alg::SparseDiffTools.SparseDiffToolsColoringAlgorithm = SparseDiffTools.GreedyD1Color(),
  kwargs...,
) where {S}
  T = eltype(S)
  output = similar(x0, ncon)
  J = compute_jacobian_sparsity(c!, output, x0)
  colors = sparse_matrix_colors(J, alg)
  jac = SparseMatrixCSC{T, Int}(J.m, J.n, J.colptr, J.rowval, T.(J.nzval))

  dx = fill!(S(undef, ncon), 0)
  cfJ = SparseDiffTools.ForwardColorJacCache(c!, x0, colorvec = colors, dx = dx, sparsity = jac)
  SDTSparseADJacobian(cfJ)
end

function get_nln_nnzj(b::SDTSparseADJacobian, nvar, ncon)
  nnz(b.cfJ.sparsity)
end

function NLPModels.jac_structure!(
  b::SDTSparseADJacobian,
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

function NLPModels.jac_coord!(
  b::SDTSparseADJacobian,
  nlp::ADModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  SparseDiffTools.forwarddiff_color_jacobian!(b.cfJ.sparsity, nlp.c!, x, b.cfJ)
  vals .= nonzeros(b.cfJ.sparsity)
  return vals
end

function NLPModels.jac_structure_residual!(
  b::SDTSparseADJacobian,
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

function NLPModels.jac_coord_residual!(
  b::SDTSparseADJacobian,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  SparseDiffTools.forwarddiff_color_jacobian!(b.cfJ.sparsity, nls.F!, x, b.cfJ)
  vals .= nonzeros(b.cfJ.sparsity)
  return vals
end

struct SDTForwardDiffADJprod{T} <: InPlaceADbackend
  tmp_in::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  tmp_out::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
end

function SDTForwardDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  tmp_in =
    Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  tmp_out =
    Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, ncon)
  return SDTForwardDiffADJprod(tmp_in, tmp_out)
end

function Jprod!(b::SDTForwardDiffADJprod, Jv, c!, x, v, ::Val)
  SparseDiffTools.auto_jacvec!(Jv, c!, x, v, b.tmp_in, b.tmp_out)
  return Jv
end

struct SDTForwardDiffADHvprod{T} <: ADBackend
  tmp_in::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  tmp_out::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
end
function SDTForwardDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  tmp_in =
    Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  tmp_out =
    Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  return SDTForwardDiffADHvprod(tmp_in, tmp_out)
end

function Hvprod!(b::SDTForwardDiffADHvprod, Hv, x, v, f, args...)
  ϕ!(dy, x; f = f) = ForwardDiff.gradient!(dy, f, x)
  SparseDiffTools.auto_hesvecgrad!(Hv, (dy, x) -> ϕ!(dy, x), x, v, b.tmp_in, b.tmp_out)
  return Hv
end

end