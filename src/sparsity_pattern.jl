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
    dcolors = nnz_colors(trilH, star_set, colors, ncolors)

Determine the coefficients in `trilH` that will be computed by a given color.

Arguments:
- `trilH::SparseMatrixCSC`: The lower triangular part of a symmetric matrix in CSC format.
- `star_set::StarSet`: A structure `StarSet` returned by the function `symmetric_coloring_detailed` of SparseMatrixColorings.jl.
- `colors::Vector{Int}`: A vector where the i-th entry represents the color assigned to the i-th column of the matrix.
- `ncolors::Int`: The number of distinct colors used in the coloring.

Output:
- `dcolors::Dict{Int, Vector{Tuple{Int, Int}}}`: A dictionary where the keys are the color indices (from 1 to `ncolors`),
and the values are vectors of tuples. Each tuple contains two integers: the first integer is the row index, and the
second integer is the index in `trilH.nzval` where the non-zero coefficient can be found.
"""
function nnz_colors(trilH, star_set, colors, ncolors)
  # We want to determine the coefficients in `trilH` that will be computed by a given color.
  # Because we exploit the symmetry, we also need to store the row index for a given coefficient
  # in the "compressed column".
  dcolors = Dict(i => Tuple{Int,Int}[] for i=1:ncolors)

  n = LinearAlgebra.checksquare(trilH)
  for j in 1:n
    for k in trilH.colptr[j]:trilH.colptr[j+1]-1
      i = trilH.rowval[k]
      l, c = symmetric_coefficient(i, j, colors, star_set)
      push!(dcolors[c], (l, k))
    end
  end

  return dcolors
end
