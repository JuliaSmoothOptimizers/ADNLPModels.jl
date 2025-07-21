function test_nlp_consistency(nlp, model; counters = true)
  nvar, ncon = model.meta.nvar, model.meta.ncon
  x = ones(nvar)
  v = 2 * ones(nvar)
  y = ones(ncon)

  # TODO: only test the backends that are defined
  if model.meta.nnln > 0
    @test jac(nlp, x) == jac(model, x)
    @test !counters || (neval_jac_nln(model) == 2)
    @test jprod(nlp, x, v) == jprod(model, x, v)
    @test !counters || (neval_jprod_nln(model) == 2)
    @test jtprod(nlp, x, y) == jtprod(model, x, y)
  end

  if (nlp isa AbstractNLSModel) && (model isa AbstractNLSModel)
    @test nlp.nls_meta.nnzj == model.nls_meta.nnzj

    nequ = model.nls_meta.nequ
    y = ones(nequ)

    @test jac_residual(nlp, x) == jac_residual(model, x)
    @test jprod_residual(nlp, x, v) == jprod_residual(model, x, v)
    @test jtprod_residual(nlp, x, y) == jtprod_residual(model, x, y)
    #@test hess_residual(nlp, x, y) == hess_residual(model, x, y)
    #for i=1:nequ
    #  @test hprod_residual(nlp, x, i, v) == hprod_residual(model, x, i, v)
    #end
  else
    @test grad(nlp, x) == grad(model, x)
    @test !counters || (neval_grad(model) == 2)
    @test hess_coord(nlp, x) == hess_coord(model, x)
    @test !counters || (neval_hess(model) == 2)
    @test hprod(nlp, x, v) == hprod(model, x, v)
    @test !counters || (neval_hprod(model) == 2)
    if model.meta.nnln > 0
      @test hess_coord(nlp, x, y) == hess_coord(model, x, y)
      @test !counters || (neval_hess(model) == 4)
      @test hprod(nlp, x, y, v) == hprod(model, x, y, v)
      @test !counters || (neval_hprod(model) == 4)
      @test ghjvprod(nlp, x, x, v) == ghjvprod(model, x, x, v)
      @test !counters || (neval_hprod(model) == 6)
      for j in model.meta.nln
        @test jth_hess(nlp, x, j) == jth_hess(model, x, j)
        @test jth_hprod(nlp, x, v, j) == jth_hprod(model, x, v, j)
      end
    end
  end
end

@testset "Test ManualNLPModel instead of AD backend" begin
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  g!(gx, x) = begin
    y1, y2 = x[1] - 1, x[2] - x[1]^2
    gx[1] = 2 * y1 - 16 * x[1] * y2
    gx[2] = 8 * y2
    return gx
  end
  hv!(hv, x, v; obj_weight = 1.0) = begin
    h11 = 2 - 16 * x[2] + 48 * x[1]^2
    h12 = -16 * x[1]
    h22 = 8.0
    hv[1] = (h11 * v[1] + h12 * v[2]) * obj_weight
    hv[2] = (h12 * v[1] + h22 * v[2]) * obj_weight
    return hv
  end
  hv!(vals, x, y, v; obj_weight = 1) = hv!(vals, x, v; obj_weight = obj_weight)

  h!(vals, x; obj_weight = 1) = begin
    vals[1] = 2 - 16 * x[2] + 48 * x[1]^2
    vals[2] = -16 * x[1]
    vals[3] = 8.0
    vals .*= obj_weight
    return vals
  end
  h!(vals, x, y; obj_weight = 1) = h!(vals, x; obj_weight = obj_weight)

  c!(cx, x) = begin
    cx[1] = x[1] + x[2]
    return cx
  end
  jv!(jv, x, v) = begin
    jv[1] = v[1] + v[2]
    return jv
  end
  jtv!(jtv, x, v) = begin
    jtv[1] = v[1]
    jtv[2] = v[1]
    return jtv
  end
  j!(vals, x) = begin
    vals[1] = 1.0
    vals[2] = 1.0
    return vals
  end

  x0 = [-1.2; 1.0]
  model = NLPModel(
    x0,
    f,
    grad = g!,
    hprod = hv!,
    hess_coord = ([1; 1; 2], [1; 2; 2], h!),
    cons = (c!, [0.0], [0.0]),
    jprod = jv!,
    jtprod = jtv!,
    jac_coord = ([1; 1], [1; 2], j!),
  )
  nlp = ADNLPModel(
    model,
    gradient_backend = model,
    hprod_backend = model,
    hessian_backend = model,
    jprod_backend = model,
    jtprod_backend = model,
    jacobian_backend = model,
    # ghjvprod_backend = model, # Not implemented for ManualNLPModels
  )

  x = rand(2)
  g = copy(x)
  y = rand(1)
  v = ones(2)

  @test grad(nlp, x) == [2 * (x[1] - 1) - 16 * x[1] * (x[2] - x[1]^2); 8 * (x[2] - x[1]^2)]
  @test hprod(nlp, x, v) == [
    (2 - 16 * x[2] + 48 * x[1]^2) * v[1] + (-16 * x[1]) * v[2]
    (-16 * x[1]) * v[1] + 8 * v[2]
  ]
  @test hess(nlp, x) == [
    2 - 16 * x[2]+48 * x[1]^2 0.0
    0.0 8.0
  ]
  @test hprod(nlp, x, y, v) == hprod(nlp, x, y, v)
  @test hess(nlp, x, y) == hess(nlp, x, y)
  @test jprod(nlp, x, v) == [2]
  @test jtprod(nlp, x, y) == [y[1]; y[1]]
  @test jac(nlp, x) == [1 1]
  @test ghjvprod(nlp, x, g, v) == [0]
end

@testset "Test mixed models with $problem" for problem in NLPModelsTest.nlp_problems
  model = eval(Meta.parse(problem))()
  nlp = ADNLPModel!(
    model,
    gradient_backend = model,
    hprod_backend = model,
    hessian_backend = model,
    jprod_backend = model,
    jtprod_backend = model,
    jacobian_backend = model,
    ghjvprod_backend = model,
  )
  test_nlp_consistency(nlp, model)

  reset!(model)
  nlp = ADNLPModel(
    model,
    gradient_backend = model,
    hprod_backend = model,
    hessian_backend = model,
    jprod_backend = model,
    jtprod_backend = model,
    jacobian_backend = model,
    ghjvprod_backend = model,
  )
  test_nlp_consistency(nlp, model)
end

@testset "Test predefined backends" begin
  f(x) = sum(x)
  function c!(cx, x)
    cx[1] = one(eltype(x))
    return cx
  end
  nvar, ncon = 2, 1
  x0 = zeros(nvar)
  lcon = ucon = zeros(1)
  adbackend = ADNLPModels.ADModelBackend(nvar, f, ncon, c!)
  nlp = ADNLPModel!(
    f,
    x0,
    c!,
    lcon,
    ucon,
    gradient_backend = adbackend.gradient_backend,
    hprod_backend = adbackend.hprod_backend,
    hessian_backend = adbackend.hessian_backend,
    jprod_backend = adbackend.jprod_backend,
    jtprod_backend = adbackend.jtprod_backend,
    jacobian_backend = adbackend.jacobian_backend,
    ghjvprod_backend = adbackend.ghjvprod_backend,
  )
  test_nlp_consistency(nlp, nlp; counters = false)
end

@testset "Test mixed NLS-models with $problem" for problem in NLPModelsTest.nls_problems
  model = eval(Meta.parse(problem))()
  nlp = ADNLSModel!(
    model,
    gradient_backend = model,
    hprod_backend = model,
    hessian_backend = model,
    jprod_backend = model,
    jtprod_backend = model,
    jacobian_backend = model,
    ghjvprod_backend = model,
    hprod_residual_backend = model,
    jprod_residual_backend = model,
    jtprod_residual_backend = model,
    jacobian_residual_backend = model,
    hessian_residual_backend = model,
  )
  test_nlp_consistency(nlp, model)

  reset!(model)
  nlp = ADNLSModel(
    model,
    gradient_backend = model,
    hprod_backend = model,
    hessian_backend = model,
    jprod_backend = model,
    jtprod_backend = model,
    jacobian_backend = model,
    ghjvprod_backend = model,
    hprod_residual_backend = model,
    jprod_residual_backend = model,
    jtprod_residual_backend = model,
    jacobian_residual_backend = model,
    hessian_residual_backend = model,
  )
  test_nlp_consistency(nlp, model)
end

@testset "Test predefined backends in NLS-models" begin
  f(x) = sum(x)
  function c!(cx, x)
    cx[1] = one(eltype(x))
    return cx
  end
  nvar, ncon, nequ = 2, 1, 2
  function F!(Fx, x)
    Fx[1] = x[1]
    Fx[2] = x[2]
    return Fx
  end
  lcon = ucon = zeros(1)
  x0 = zeros(nvar)
  adbackend = ADNLPModels.ADModelNLSBackend(nvar, F!, nequ, ncon, c!)
  nlp = ADNLSModel!(
    F!,
    x0,
    nequ,
    c!,
    lcon,
    ucon,
    jprod_backend = adbackend.jprod_backend,
    jtprod_backend = adbackend.jtprod_backend,
    jacobian_backend = adbackend.jacobian_backend,
    jprod_residual_backend = adbackend.jprod_residual_backend,
    jtprod_residual_backend = adbackend.jtprod_residual_backend,
    jacobian_residual_backend = adbackend.jacobian_residual_backend,
  )
  test_nlp_consistency(nlp, nlp; counters = false)
end
