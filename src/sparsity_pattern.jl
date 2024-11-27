export get_sparsity_pattern

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
    S = get_sparsity_pattern(model::ADModel, derivative::Symbol)

Retrieve the sparsity pattern of a Jacobian or Hessian from an `ADModel`.
For the Hessian, only the lower triangular part of its sparsity pattern is returned.
The user can reconstruct the upper triangular part by exploiting symmetry.

To compute the sparsity pattern, the model must use a sparse backend.
Supported backends include `SparseADJacobian`, `SparseADHessian`, and `SparseReverseADHessian`.

#### Input arguments

* `model`: An automatic differentiation model (either `AbstractADNLPModel` or `AbstractADNLSModel`).
* `derivative`: The type of derivative for which the sparsity pattern is needed. The supported values are `:jacobian`, `:hessian`, `:jacobian_residual` and `:hessian_residual`.

#### Output argument

* `S`: A sparse matrix of type `SparseMatrixCSC{Bool,Int}` indicating the sparsity pattern of the requested derivative.
"""
function get_sparsity_pattern(model::ADModel, derivative::Symbol)
  get_sparsity_pattern(model, Val(derivative))
end

function get_sparsity_pattern(model::ADModel, ::Val{:jacobian})
  backend = model.adbackend.jacobian_backend
  validate_sparse_backend(
    backend,
    Union{SparseADJacobian, SparseEnzymeADJacobian},
    "Jacobian",
  )
  m = model.meta.ncon
  n = model.meta.nvar
  colptr = backend.colptr
  rowval = backend.rowval
  nnzJ = length(rowval)
  nzval = ones(Bool, nnzJ)
  SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function get_sparsity_pattern(model::ADModel, ::Val{:hessian})
  backend = model.adbackend.hessian_backend
  validate_sparse_backend(backend, Union{SparseADHessian, SparseReverseADHessian}, "Hessian")
  n = model.meta.nvar
  colptr = backend.colptr
  rowval = backend.rowval
  nnzH = length(rowval)
  nzval = ones(Bool, nnzH)
  SparseMatrixCSC(n, n, colptr, rowval, nzval)
end

function get_sparsity_pattern(model::AbstractADNLSModel, ::Val{:jacobian_residual})
  backend = model.adbackend.jacobian_residual_backend
  validate_sparse_backend(
    backend,
    Union{SparseADJacobian, SparseEnzymeADJacobian},
    "Jacobian of the residual",
  )
  m = model.nls_meta.nequ
  n = model.meta.nvar
  colptr = backend.colptr
  rowval = backend.rowval
  nnzJ = length(rowval)
  nzval = ones(Bool, nnzJ)
  SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function get_sparsity_pattern(model::AbstractADNLSModel, ::Val{:hessian_residual})
  backend = model.adbackend.hessian_residual_backend
  validate_sparse_backend(
    backend,
    Union{SparseADHessian, SparseReverseADHessian},
    "Hessian of the residual",
  )
  n = model.meta.nvar
  colptr = backend.colptr
  rowval = backend.rowval
  nnzH = length(rowval)
  nzval = ones(Bool, nnzH)
  SparseMatrixCSC(n, n, colptr, rowval, nzval)
end

function validate_sparse_backend(
  backend::B,
  expected_type,
  derivative_name::String,
) where {B <: ADBackend}
  if !(backend isa expected_type)
    error("The current backend $B doesn't compute a sparse $derivative_name.")
  end
end
