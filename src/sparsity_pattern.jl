"""
    compute_jacobian_sparsity(c, x0)
    compute_jacobian_sparsity(c!, cx, x0)

Return a sparse matrix.
"""
function compute_jacobian_sparsity end

function compute_jacobian_sparsity(c, x0)
  detector = SparseConnectivityTracer.TracerSparsityDetector()  # replaceable
  S = SparseConnectivityTracer.jacobian_pattern(c, x0, detector)
  return S
end

function compute_jacobian_sparsity(c!, cx, x0)
  detector = SparseConnectivityTracer.TracerSparsityDetector()  # replaceable
  S = ADTypes.jacobian_sparsity(c!, cx, x0, detector)
  return S
end

"""
    compute_hessian_sparsity(f, nvar, c!, ncon)

Return a sparse matrix.
"""
function compute_hessian_sparsity(f, nvar, c!, ncon)
  function lagrangian(x)
    if ncon == 0
      return f(x)
    else
      cx = zeros(eltype(x), ncon)
      y0 = rand(ncon)
      return f(x) + dot(c!(cx, x), y0)
    end
  end

  detector = SparseConnectivityTracer.TracerSparsityDetector()  # replaceable
  x0 = rand(nvar)
  S = ADTypes.hessian_sparsity(lagrangian, x0, detector)
  return S
end
