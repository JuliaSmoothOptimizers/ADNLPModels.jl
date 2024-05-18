"""
    compute_jacobian_sparsity(c!, cx, x0)

Return a sparse matrix.
"""
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
  detector = SparseConnectivityTracer.TracerSparsityDetector()  # replaceable
  function lagrangian(x)
    cx = zeros(eltype(x), ncon)
    c!(cx, x)
    return f(x) + dot(rand(ncon), cx)
  end
  S = ADTypes.hessian_sparsity(lagrangian, rand(nvar), detector)
  return S
end
