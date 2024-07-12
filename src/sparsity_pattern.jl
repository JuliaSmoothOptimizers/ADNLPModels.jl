"""
    compute_jacobian_sparsity(c, x0; detector)
    compute_jacobian_sparsity(c!, cx, x0; detector)

Return a sparse boolean matrix that represents the adjacency matrix of the Jacobian of c(x).
"""
function compute_jacobian_sparsity end

function compute_jacobian_sparsity(
  c,
  x0;
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
)
  S = ADTypes.jacobian_sparsity(c, x0, detector)
  return S
end

function compute_jacobian_sparsity(
  c!,
  cx,
  x0;
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
)
  S = ADTypes.jacobian_sparsity(c!, cx, x0, detector)
  return S
end

"""
    compute_hessian_sparsity(f, nvar, c!, ncon; detector)

Return a sparse boolean matrix that represents the adjacency matrix of the Hessian of f(x) + λᵀc(x).
"""
function compute_hessian_sparsity(
  f,
  nvar,
  c!,
  ncon;
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
)
  function lagrangian(x)
    if ncon == 0
      return f(x)
    else
      cx = zeros(eltype(x), ncon)
      y0 = rand(ncon)
      return f(x) + dot(c!(cx, x), y0)
    end
  end

  x0 = rand(nvar)
  S = ADTypes.hessian_sparsity(lagrangian, x0, detector)
  return S
end

"""
    dcolors = nnz_colors(trilH, colors, ncolors)

Determine the coefficients in `trilH` that will be computed by a given color.
This function leverages the symmetry of the matrix `H` and also stores the row index for a
given coefficient in the "compressed column".

# Arguments
- `H::SparseMatrixCSC`: A sparse matrix in CSC format.
- `trilH::SparseMatrixCSC`: The lower triangular part of `H` in CSC format.
- `colors::Vector{Int}`: A vector where the i-th entry represents the color assigned to the i-th column of the matrix.
- `ncolors::Int`: The number of distinct colors used in the coloring.

# Output
- `dcolors::Dict{Int, Vector{Tuple{Int, Int}}}`: A dictionary where the keys are the color indices (from 1 to `ncolors`),
and the values are vectors of tuples. Each tuple contains two integers: the first integer is the row index, and the
second integer is the index in `trilH.nzval` where the non-zero coefficient can be found.
"""
function nnz_colors(H, trilH, colors, ncolors)
  # We want to determine the coefficients in `trilH` that will be computed by a given color.
  # Because we exploit the symmetry, we also need to store the row index for a given coefficient
  # in the "compressed column".
  dcolors = Dict(i => Tuple{Int,Int}[] for i=1:ncolors)

  n = LinearAlgebra.checksquare(trilH)
  for j in 1:n
    for k in trilH.colptr[j]:trilH.colptr[j+1]-1
      i = trilH.rowval[k]

      # Should we use c[i] or c[j] for this coefficient H[i,j]?
      ci = colors[i]
      cj = colors[j]

      if i == j
        # H[i,j] is a coefficient of the diagonal
        push!(dcolors[ci], (j,k))
      else # i > j
        if is_only_color_in_row(H, i, j, n, colors, ci)
          # column i is the only column of its color c[i] with a non-zero coefficient in row j
          push!(dcolors[ci], (j,k))
        else
          # column j is the only column of its color c[j] with a non-zero coefficient in row i
          # it is ensured by the star coloring used in `symmetric_coloring`.
          push!(dcolors[cj], (i,k))
        end
      end
    end
  end

  return dcolors
end

"""
    flag = is_only_color_in_row(H, i, j, n, colors, ci)

This function returns `true` if the column `i` is the only column of color `ci` with a non-zero coefficient in row `j`.
It returns `false` otherwise.
"""
function is_only_color_in_row(H, i, j, n, colors, ci)
  column = 1
  flag = true
  while flag && column ≤ n
    # We want to check that all columns (excpect the i-th one)
    # with color ci doesn't have a non-zero coefficient in row j
    if (column != i) && (colors[column] == ci)
      k = H.colptr[column]
      while (k ≤ H.colptr[column+1]-1) && flag
        row = H.rowval[k]
        if row == j
          # We found a coefficient at row j in a column of color ci
          flag = false
        else # row != j
          k += 1
        end
      end
    end
    column += 1
  end
  return flag
end
