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
  alg::SparseDiffTools.SparseDiffToolsColoringAlgorithm = SparseDiffTools.GreedyD1Color(),
  kwargs...,
) where {T}
  output = similar(x0, ncon)
  J = Symbolics.jacobian_sparsity(c!, output, x0)
  colors = matrix_colors(J, alg)
  d = BitVector(undef, nvar)
  ncolors = maximum(colors)

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

function jac_structure!(
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

function jac_coord!(b::SparseADJacobian, nlp::ADModel, x::AbstractVector, vals::AbstractVector)
  sparse_jac_coord!(nlp.c!, b, x, vals)
  return vals
end

function jac_structure_residual!(
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

function jac_coord_residual!(
  b::SparseADJacobian,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  sparse_jac_coord!(nls.F!, b, x, vals)
  return vals
end

## ----- Symbolics -----

struct SparseSymbolicsADJacobian{T} <: ADBackend
  nnzj::Int
  rows::Vector{Int}
  cols::Vector{Int}
  cfJ::T
end

function SparseSymbolicsADJacobian(nvar, f, ncon, c!; kwargs...)
  @variables xs[1:nvar] out[1:ncon]
  wi = Symbolics.scalarize(xs)
  wo = Symbolics.scalarize(out)
  fun = c!(wo, wi)
  J = Symbolics.jacobian_sparsity(c!, wo, wi)
  rows, cols, _ = findnz(J)
  vals = Symbolics.sparsejacobian_vals(fun, wi, rows, cols)
  nnzj = length(vals)
  # cfJ is a Tuple{Expr, Expr}, cfJ[2] is the in-place function
  # that we need to update a vector `vals` with the nonzeros of Jc(x).
  cfJ = Symbolics.build_function(vals, wi, expression = Val{false})
  SparseSymbolicsADJacobian(nnzj, rows, cols, cfJ[2])
end

function get_nln_nnzj(b::SparseSymbolicsADJacobian, nvar, ncon)
  b.nnzj
end

function jac_structure!(
  b::SparseSymbolicsADJacobian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end

function jac_coord!(
  b::SparseSymbolicsADJacobian,
  nlp::ADModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  @eval $(b.cfJ)($vals, $x)
  return vals
end

function jac_structure_residual!(
  b::SparseSymbolicsADJacobian,
  nls::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end

function jac_coord_residual!(
  b::SparseSymbolicsADJacobian,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  @eval $(b.cfJ)($vals, $x)
  return vals
end
