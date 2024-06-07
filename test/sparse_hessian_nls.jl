list_sparse_hess_backend = (
  (ADNLPModels.SparseADHessian, Dict()),
  (ADNLPModels.ForwardDiffADHessian, Dict()),
)

dt = (Float32, Float64)

@testset "Basic Hessian of residual derivative with backend=$(backend) and T=$(T)" for T in dt,
  (backend, kw) in list_sparse_hess_backend

  F!(Fx, x) = begin
    Fx[1] = x[1] - 1
    Fx[2] = 10 * (x[2] - x[1]^2)
    Fx[3] = x[2] + 1
    Fx
  end
  x0 = T[-1.2; 1.0]
  nvar = 2
  nequ = 3
  nls = ADNLPModels.ADNLSModel!(F!, x0, 3, hessian_residual_backend = backend; kw...)

  x = rand(T, nvar)
  v = rand(T, nequ)
  rows, cols = zeros(Int, nls.nls_meta.nnzh), zeros(Int, nls.nls_meta.nnzh)
  vals = zeros(T, nls.nls_meta.nnzh)
  hess_structure_residual!(nls, rows, cols)
  hess_coord_residual!(nls, x, v, vals)
  @test eltype(vals) == T
  H = sparse(rows, cols, vals, nvar, nvar)
  # @test H == []

  # Test also the implementation of the backends
  b = nls.adbackend.hessian_residual_backend
  obj_weight = 0.5
  @test nls.nls_meta.nnzh == ADNLPModels.get_nln_nnzh(b, nvar)
  ADNLPModels.hess_structure_residual!(b, nls, rows, cols)
  ADNLPModels.hess_coord_residual!(b, nls, x, y, obj_weight, vals)
  @test eltype(vals) == T
  H = sparse(rows, cols, vals, nvar, nvar)
  # @test H == []

  nls = ADNLPModels.ADNLSModel!(F!, x0, 3, matrix_free = true; kw...)
  @test nls.adbackend.hessian_backend isa ADNLPModels.EmptyADbackend
  @test nls.adbackend.hessian_residual_backend isa ADNLPModels.EmptyADbackend
end
