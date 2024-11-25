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
    S = get_sparsity_pattern(model::ADModel, derivate::Symbol)

Retrieve the sparsity pattern of a Jacobian or Hessian from an `ADModel`.
For the Hessian, only the lower triangular part of its sparsity pattern is returned.
The user can reconstruct the upper triangular part by exploiting symmetry.

To compute the sparsity pattern, the model must use a sparse backend.
Supported backends include `SparseADJacobian`, `SparseADHessian`, and `SparseReverseADHessian`.

#### Input arguments

* `model`: An automatic differentiation model (either `AbstractADNLPModel` or `AbstractADNLSModel`).
* `derivate`: The type of derivative for which the sparsity pattern is needed. The supported values are `:jacobian`, `:hessian`, `:jacobian_residual` and `:hessian_residual`.

#### Output argument

* `S`: A sparse matrix of type `SparseMatrixCSC{Bool,Int}` indicating the sparsity pattern of the requested derivative.
"""
function get_sparsity_pattern(model::ADModel, derivate::Symbol)
  if (derivate != :jacobian) && (derivate != :hessian)
    if model isa AbstractADNLPModel
      error("The only supported sparse derivates for an AbstractADNLPModel are `:jacobian` and `:hessian`.")
    elseif (derivate != :jacobian_residual) && (derivate != :hessian_resiual)
      error("The only supported sparse derivates for an AbstractADNLSModel are `:jacobian`, `:jacobian_residual`, `:hessian` and `:hessian_resiual`.")
    end
  end
  if (derivate == :jacobian) || (derivate == :jacobian_residual)
    backend = derivate == :jacobian ? model.adbackend.jacobian_backend : model.adbackend.jacobian_residual_backend
    if backend isa SparseADJacobian
      m = model.meta.nvar
      n = derivate == :jacobian ? model.meta.ncon : model.nls_meta.nequ
      colptr = backend.colptr
      rowval = backend.rowval
      nnzJ = length(rowval)
      nzval = ones(Bool, nnzJ)
      J = SparseMatrixCSC(m, n, colptr, rowval, nzval)
      return J
    else
      B = typeof(backend)
      error("The current backend ($B) doesn't compute a sparse Jacobian.")
    end
  end
  if (derivate == :hessian) || (derivate == :hessian_residual)
    backend = derivate == :hessian ? model.adbackend.hessian_backend : model.adbackend.hessian_residual_backend
    if (backend isa SparseADHessian) || (backend isa SparseReverseADHessian)
      n = model.meta.nvar
      colptr = backend.colptr
      rowval = backend.rowval
      nnzH = length(rowval)
      nzval = ones(Bool, nnzH)
      H = SparseMatrixCSC(n, n, colptr, rowval, nzval)
      return H
    else
      B = typeof(backend)
      error("The current backend ($B) doesn't compute a sparse Hessian.")
    end
  end
end
