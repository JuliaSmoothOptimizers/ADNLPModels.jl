if test_enzyme
  list_sparse_jac_backend = ((ADNLPModels.SparseEnzymeADJacobian, Dict()),)
else
  list_sparse_jac_backend = ((ADNLPModels.SparseADJacobian, Dict()),
                             (ADNLPModels.ForwardDiffADJacobian, Dict()))
end

dt = (Float32, Float64)

@testset "Basic Jacobian of residual derivative with backend=$(backend) and T=$(T)" for T in dt,
  (backend, kw) in list_sparse_jac_backend

  F!(Fx, x) = begin
    Fx[1] = x[1] - 1
    Fx[2] = 10 * (x[2] - x[1]^2)
    Fx[3] = x[2] + 1
    Fx
  end
  x0 = T[-1.2; 1.0]
  nvar = 2
  nequ = 3
  nls = ADNLPModels.ADNLSModel!(F!, x0, 3, jacobian_residual_backend = backend; kw...)

  x = rand(T, 2)
  rows, cols = zeros(Int, nls.nls_meta.nnzj), zeros(Int, nls.nls_meta.nnzj)
  vals = zeros(T, nls.nls_meta.nnzj)
  jac_structure_residual!(nls, rows, cols)
  jac_coord_residual!(nls, x, vals)
  @test eltype(vals) == T
  J = sparse(rows, cols, vals, nequ, nvar)
  @test J == [
    1 0
    -20*x[1] 10
    0 1
  ]

  # Test also the implementation of the backends
  b = nls.adbackend.jacobian_residual_backend
  @test nls.nls_meta.nnzj == ADNLPModels.get_nln_nnzj(b, nvar, nequ)
  ADNLPModels.jac_structure_residual!(b, nls, rows, cols)
  ADNLPModels.jac_coord_residual!(b, nls, x, vals)
  @test eltype(vals) == T
  J = sparse(rows, cols, vals, nequ, nvar)
  @test J == [
    1 0
    -20*x[1] 10
    0 1
  ]

  if backend != ADNLPModels.ForwardDiffADJacobian
    J_sp = get_sparsity_pattern(nls, :jacobian_residual)
    @test J_sp == SparseMatrixCSC{Bool, Int}([
      1 0
      1 1
      0 1
    ])
  end

  nls = ADNLPModels.ADNLSModel!(F!, x0, 3, matrix_free = true; kw...)
  @test nls.adbackend.jacobian_backend isa ADNLPModels.EmptyADbackend
  @test nls.adbackend.jacobian_residual_backend isa ADNLPModels.EmptyADbackend
end
