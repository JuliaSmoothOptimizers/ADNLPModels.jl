list_sparse_hess_backend = (
  (ADNLPModels.ForwardDiffADHessian, Dict()),
  (ADNLPModels.SparseADHessian, Dict()),
)
dt = (Float32, Float64)
@testset "Basic Hessian derivative with backend=$(backend) and T=$(T)" for T in dt,
  (backend, kw) in list_sparse_hess_backend

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
    x -> x[1]*x[2]^2 + x[1]^2*x[2],
    x0,
    c!,
    zeros(T, ncon),
    zeros(T, ncon),
    hessian_backend = backend;
    kw...,
  )

  x = rand(T, 2)
  y = rand(T, 3)
  rows, cols = zeros(Int, nlp.meta.nnzh), zeros(Int, nlp.meta.nnzh)
  vals = zeros(T, nlp.meta.nnzh)
  hess_structure!(nlp, rows, cols)
  hess_coord!(nlp, x, vals)
  @test eltype(vals) == T
  H = sparse(rows, cols, vals, nvar, nvar)
  @test H == [2*x[2] 0; 2*(x[1]+x[2]) 2*x[1]]

  # Test also the implementation of the backends
  b = nlp.adbackend.hessian_backend
  obj_weight = 0.5
  @test nlp.meta.nnzh == ADNLPModels.get_nln_nnzh(b, nvar)
  ADNLPModels.hess_structure!(b, nlp, rows, cols)
  ADNLPModels.hess_coord!(b, nlp, x, obj_weight, vals)
  @test eltype(vals) == T
  H = sparse(rows, cols, vals, nvar, nvar)
  @test H == [x[2] 0; x[1]+x[2] x[1]]
  ADNLPModels.hess_coord!(b, nlp, x, y, obj_weight, vals)
  @test eltype(vals) == T
  H = sparse(rows, cols, vals, nvar, nvar)
  @test H == [x[2] 0; x[1]+x[2] x[1]] + y[2] * [-20 0; 0 0]
end
