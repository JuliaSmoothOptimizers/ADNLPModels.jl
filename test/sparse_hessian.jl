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
    x -> x[1] * x[2]^2 + x[1]^2 * x[2],
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
  @test H == [2*x[2] 0; 2*(x[1] + x[2]) 2*x[1]]

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

  if (backend == ADNLPModels.SparseADHessian) || (backend == ADNLPModels.SparseReverseADHessian)
    H_sp = get_sparsity_pattern(nlp, :hessian)
    @test H_sp == SparseMatrixCSC{Bool, Int}([
      1 0
      1 1
    ])
  end

  nlp = ADNLPModel!(
    x -> x[1] * x[2]^2 + x[1]^2 * x[2],
    x0,
    c!,
    zeros(T, ncon),
    zeros(T, ncon),
    matrix_free = true;
    kw...,
  )
  @test nlp.adbackend.hessian_backend isa ADNLPModels.EmptyADbackend

  n = 4
  x = ones(T, 4)
  nlp = ADNLPModel(
    x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
    x,
    hessian_backend = backend,
    name = "Extended Rosenbrock",
  )
  @test hess(nlp, x) == T[802 -400 0 0; -400 1002 -400 0; 0 -400 1002 -400; 0 0 -400 200]

  x = ones(T, 2)
  nlp = ADNLPModel(x -> x[1]^2 + x[1] * x[2], x, hessian_backend = backend)
  @test hess(nlp, x) == T[2 1; 1 0]

  nlp = ADNLPModel(
    x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
    x,
    name = "Extended Rosenbrock",
    matrix_free = true,
  )
  @test nlp.adbackend.hessian_backend isa ADNLPModels.EmptyADbackend
end
