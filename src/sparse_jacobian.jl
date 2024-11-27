struct SparseADJacobian{Tag, R, T, C, S} <: ADBackend
  nvar::Int
  ncon::Int
  rowval::Vector{Int}
  colptr::Vector{Int}
  nzval::Vector{R}
  result_coloring::C
  compressed_jacobian::S
  seed::BitVector
  z::Vector{ForwardDiff.Dual{Tag, T, 1}}
  cz::Vector{ForwardDiff.Dual{Tag, T, 1}}
end

function SparseADJacobian(
  nvar,
  f,
  ncon,
  c!;
  x0::AbstractVector = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:direct}(),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  kwargs...,
)
  output = similar(x0, ncon)
  J = compute_jacobian_sparsity(c!, output, x0, detector = detector)
  SparseADJacobian(nvar, f, ncon, c!, J; x0, coloring_algorithm, kwargs...)
end

function SparseADJacobian(
  nvar,
  f,
  ncon,
  c!,
  J::SparseMatrixCSC{Bool, Int};
  x0::AbstractVector{T} = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:direct}(),
  kwargs...,
) where {T}
  # We should support :row and :bidirectional in the future
  problem = ColoringProblem{:nonsymmetric, :column}()
  result_coloring = coloring(J, problem, coloring_algorithm, decompression_eltype = T)

  rowval = J.rowval
  colptr = J.colptr
  nzval = T.(J.nzval)
  compressed_jacobian = similar(x0, ncon)
  seed = BitVector(undef, nvar)

  tag = ForwardDiff.Tag{typeof(c!), T}
  z = Vector{ForwardDiff.Dual{tag, T, 1}}(undef, nvar)
  cz = similar(z, ncon)

  SparseADJacobian(
    nvar,
    ncon,
    rowval,
    colptr,
    nzval,
    result_coloring,
    compressed_jacobian,
    seed,
    z,
    cz,
  )
end

function sparse_jac_coord!(
  ℓ!::Function,
  b::SparseADJacobian{Tag},
  x::AbstractVector,
  vals::AbstractVector,
) where {Tag}
  # SparseMatrixColorings.jl requires a SparseMatrixCSC for the decompression
  A = SparseMatrixCSC(b.ncon, b.nvar, b.colptr, b.rowval, b.nzval)

  groups = column_groups(b.result_coloring)
  for (icol, cols) in enumerate(groups)
    # Update the seed
    b.seed .= false
    for col in cols
      b.seed[col] = true
    end

    map!(ForwardDiff.Dual{Tag}, b.z, x, b.seed)  # x + ε * v
    ℓ!(b.cz, b.z)  # c!(cz, x + ε * v)
    ForwardDiff.extract_derivative!(Tag, b.compressed_jacobian, b.cz)  # ∇c!(cx, x)ᵀv

    # Update the columns of the Jacobian that have the color `icol`
    decompress_single_color!(A, b.compressed_jacobian, icol, b.result_coloring)
  end
  vals .= b.nzval
  return vals
end

struct SparseEnzymeADJacobian{R, C, S} <: ADBackend
  nvar::Int
  ncon::Int
  rowval::Vector{Int}
  colptr::Vector{Int}
  nzval::Vector{R}
  result_coloring::C
  compressed_jacobian::S
  v::Vector{R}
  buffer::Vector{R}
end

function SparseEnzymeADJacobian(
  nvar,
  f,
  ncon,
  c!;
  x0::AbstractVector = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:direct}(),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  kwargs...,
)
  output = similar(x0, ncon)
  J = compute_jacobian_sparsity(c!, output, x0, detector = detector)
  SparseEnzymeADJacobian(nvar, f, ncon, c!, J; x0, coloring_algorithm, kwargs...)
end

function SparseEnzymeADJacobian(
  nvar,
  f,
  ncon,
  c!,
  J::SparseMatrixCSC{Bool, Int};
  x0::AbstractVector{T} = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:direct}(),
  kwargs...,
) where {T}
  # We should support :row and :bidirectional in the future
  problem = ColoringProblem{:nonsymmetric, :column}()
  result_coloring = coloring(J, problem, coloring_algorithm, decompression_eltype = T)

  rowval = J.rowval
  colptr = J.colptr
  nzval = T.(J.nzval)
  compressed_jacobian = similar(x0, ncon)
  v = similar(x0)
  buffer = zeros(T, ncon)

  SparseEnzymeADJacobian(
    nvar,
    ncon,
    rowval,
    colptr,
    nzval,
    result_coloring,
    compressed_jacobian,
    v,
    buffer,
  )
end

function sparse_jac_coord!(
  c!::Function,
  b::SparseEnzymeADJacobian,
  x::AbstractVector,
  vals::AbstractVector,
)
  # SparseMatrixColorings.jl requires a SparseMatrixCSC for the decompression
  A = SparseMatrixCSC(b.ncon, b.nvar, b.colptr, b.rowval, b.nzval)

  groups = column_groups(b.result_coloring)
  for (icol, cols) in enumerate(groups)
    # Update the seed
    b.v .= 0
    for col in cols
      b.v[col] = 1
    end

    # b.compressed_jacobian is just a vector Jv here
    # We don't use the vector mode
    Enzyme.autodiff(Enzyme.Forward, Const(c!), Duplicated(b.buffer, b.compressed_jacobian), Duplicated(x, b.v))

    # Update the columns of the Jacobian that have the color `icol`
    decompress_single_color!(A, b.compressed_jacobian, icol, b.result_coloring)
  end
  vals .= b.nzval
  return vals
end

function get_nln_nnzj(b::Union{SparseADJacobian, SparseEnzymeADJacobian}, nvar, ncon)
  length(b.rowval)
end

function NLPModels.jac_structure!(
  b::Union{SparseADJacobian, SparseEnzymeADJacobian},
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

function NLPModels.jac_coord!(
  b::Union{SparseADJacobian, SparseEnzymeADJacobian},
  nlp::ADModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  sparse_jac_coord!(nlp.c!, b, x, vals)
  return vals
end

function NLPModels.jac_structure_residual!(
  b::Union{SparseADJacobian, SparseEnzymeADJacobian},
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
  b::Union{SparseADJacobian, SparseEnzymeADJacobian},
  nls::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  sparse_jac_coord!(nls.F!, b, x, vals)
  return vals
end
