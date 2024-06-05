"""
    compute_jacobian_sparsity(c, x0; detector)
    compute_jacobian_sparsity(c!, cx, x0; detector)

Return a sparse boolean matrix that represents the adjacency matrix of the Jacobian of c(x).
"""
function compute_jacobian_sparsity end

function compute_jacobian_sparsity(c, x0; detector::AbstractSparsityDetector=TracerSparsityDetector())
  S = ADTypes.jacobian_sparsity(c, x0, detector)
  return S
end

function compute_jacobian_sparsity(c!, cx, x0; detector::AbstractSparsityDetector=TracerSparsityDetector())
  S = ADTypes.jacobian_sparsity(c!, cx, x0, detector)
  return S
end

"""
    compute_hessian_sparsity(f, nvar, c!, ncon; detector)

Return a sparse boolean matrix that represents the adjacency matrix of the Hessian of f(x) + λᵀc(x).
"""
function compute_hessian_sparsity(f, nvar, c!, ncon; detector::AbstractSparsityDetector=TracerSparsityDetector())
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
