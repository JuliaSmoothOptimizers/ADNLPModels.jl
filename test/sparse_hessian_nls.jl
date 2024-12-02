if test_enzyme
  list_sparse_hess_backend = (
    ( ADNLPModels.SparseEnzymeADHessian,
      Dict(:coloring_algorithm => GreedyColoringAlgorithm{:direct}()),
    ),
    (
      ADNLPModels.SparseEnzymeADHessian,
      Dict(:coloring_algorithm => GreedyColoringAlgorithm{:substitution}()),
    ),
  )
else
  list_sparse_hess_backend = (
    (ADNLPModels.SparseADHessian, Dict(:coloring_algorithm => GreedyColoringAlgorithm{:direct}())),
    (
      ADNLPModels.SparseADHessian,
      Dict(:coloring_algorithm => GreedyColoringAlgorithm{:substitution}()),
    ),
    (
      ADNLPModels.SparseReverseADHessian,
      Dict(:coloring_algorithm => GreedyColoringAlgorithm{:direct}()),
    ),
    (
      ADNLPModels.SparseReverseADHessian,
      Dict(:coloring_algorithm => GreedyColoringAlgorithm{:substitution}()),
    ),
    (ADNLPModels.ForwardDiffADHessian, Dict()),
  )
end

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
  H = Symmetric(sparse(rows, cols, vals, nvar, nvar), :L)
  @test H == [-20*v[2] 0; 0 0]

  # Test also the implementation of the backends
  b = nls.adbackend.hessian_residual_backend
  @test nls.nls_meta.nnzh == ADNLPModels.get_nln_nnzh(b, nvar)
  ADNLPModels.hess_structure_residual!(b, nls, rows, cols)
  ADNLPModels.hess_coord_residual!(b, nls, x, v, vals)
  @test eltype(vals) == T
  H = Symmetric(sparse(rows, cols, vals, nvar, nvar), :L)
  @test H == [-20*v[2] 0; 0 0]

  if backend != ADNLPModels.ForwardDiffADHessian
    H_sp = get_sparsity_pattern(nls, :hessian_residual)
    @test H_sp == SparseMatrixCSC{Bool, Int}([
      1 0
      0 0
    ])
  end

  nls = ADNLPModels.ADNLSModel!(F!, x0, 3, matrix_free = true; kw...)
  @test nls.adbackend.hessian_backend isa ADNLPModels.EmptyADbackend
  @test nls.adbackend.hessian_residual_backend isa ADNLPModels.EmptyADbackend
end
