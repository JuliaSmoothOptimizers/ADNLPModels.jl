struct SparseADJacobian{T, Tag, S} <: ADBackend
  d::BitVector
  rowval::Vector{Int}
  colptr::Vector{Int}
  colors::Vector{Int}
  ncolors::Int
  z::Vector{ForwardDiff.Dual{Tag, T, 1}}
  cz::Vector{ForwardDiff.Dual{Tag, T, 1}}
  res::S
end

function SparseADJacobian(
  nvar,
  f,
  ncon,
  c!;
  x0::AbstractVector{T} = rand(nvar),
  coloring::AbstractColoringAlgorithm = GreedyColoringAlgorithm(),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  kwargs...,
) where {T}
  output = similar(x0, ncon)
  J = compute_jacobian_sparsity(c!, output, x0, detector = detector)

  # TODO: use ADTypes.row_coloring instead if you have the right decompression and some heuristic recommends it
  colors = ADTypes.column_coloring(J, coloring)
  ncolors = maximum(colors)

  d = BitVector(undef, nvar)

  rowval = J.rowval
  colptr = J.colptr

  tag = ForwardDiff.Tag{typeof(c!), T}

  z = Vector{ForwardDiff.Dual{tag, T, 1}}(undef, nvar)
  cz = similar(z, ncon)
  res = similar(x0, ncon)

  SparseADJacobian(d, rowval, colptr, colors, ncolors, z, cz, res)
end

function get_nln_nnzj(b::SparseADJacobian, nvar, ncon)
  length(b.rowval)
end

function NLPModels.jac_structure!(
  b::SparseADJacobian,
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
  ℓ!::Function,
  b::SparseADJacobian{T, Tag},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, Tag}
  nvar = length(x)
  for icol = 1:(b.ncolors)
    b.d .= (b.colors .== icol)
    map!(ForwardDiff.Dual{Tag}, b.z, x, b.d) # x + ε * v
    ℓ!(b.cz, b.z) # c!(cz, x + ε * v)
    ForwardDiff.extract_derivative!(Tag, b.res, b.cz) # ∇c!(cx, x)ᵀv
    for j = 1:nvar
      if b.colors[j] == icol
        for k = b.colptr[j]:(b.colptr[j + 1] - 1)
          i = b.rowval[k]
          vals[k] = b.res[i]
        end
      end
    end
  end
  return vals
end

function NLPModels.jac_coord!(
  b::SparseADJacobian,
  nlp::ADModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  sparse_jac_coord!(nlp.c!, b, x, vals)
  return vals
end

function NLPModels.jac_structure_residual!(
  b::SparseADJacobian,
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
  b::SparseADJacobian,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  sparse_jac_coord!(nls.F!, b, x, vals)
  return vals
end
