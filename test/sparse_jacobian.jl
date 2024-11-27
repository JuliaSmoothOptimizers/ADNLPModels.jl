list_sparse_jac_backend =
  ((ADNLPModels.SparseADJacobian, Dict()), (ADNLPModels.ForwardDiffADJacobian, Dict()))

dt = (Float32, Float64)

@testset "Basic Jacobian derivative with backend=$(backend) and T=$(T)" for T in dt,
  (backend, kw) in list_sparse_jac_backend

  c!(cx, x) = begin
    cx[1] = x[1] - 1
    cx[2] = 10 * (x[2] - x[1]^2)
    cx[3] = x[2] + 1
    cx
  end
  x0 = T[-1.2; 1.0]
  nvar = 2
  ncon = 3
  nlp = ADNLPModel!(
    x -> sum(x),
    x0,
    c!,
    zeros(T, ncon),
    zeros(T, ncon),
    jacobian_backend = backend;
    kw...,
  )

  x = rand(T, 2)
  rows, cols = zeros(Int, nlp.meta.nln_nnzj), zeros(Int, nlp.meta.nln_nnzj)
  vals = zeros(T, nlp.meta.nln_nnzj)
  jac_nln_structure!(nlp, rows, cols)
  jac_nln_coord!(nlp, x, vals)
  @test eltype(vals) == T
  J = sparse(rows, cols, vals, ncon, nvar)
  @test J == [
    1 0
    -20*x[1] 10
    0 1
  ]

  # Test also the implementation of the backends
  b = nlp.adbackend.jacobian_backend
  @test nlp.meta.nnzj == ADNLPModels.get_nln_nnzj(b, nvar, ncon)
  ADNLPModels.jac_structure!(b, nlp, rows, cols)
  ADNLPModels.jac_coord!(b, nlp, x, vals)
  @test eltype(vals) == T
  J = sparse(rows, cols, vals, ncon, nvar)
  @test J == [
    1 0
    -20*x[1] 10
    0 1
  ]

  if backend == ADNLPModels.SparseADJacobian
    J_sp = get_sparsity_pattern(nlp, :jacobian)
    @test J_sp == SparseMatrixCSC{Bool, Int}(
      [ 1 0 ;
        1 1 ;
        0 1 ]
    )
  end

  nlp = ADNLPModel!(x -> sum(x), x0, c!, zeros(T, ncon), zeros(T, ncon), matrix_free = true; kw...)
  @test nlp.adbackend.jacobian_backend isa ADNLPModels.EmptyADbackend
end
