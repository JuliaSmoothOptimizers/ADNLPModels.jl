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
  hv!(vals, x, y, v; obj_weight=1) = hv!(vals, x, v; obj_weight=obj_weight)

  h!(vals, x; obj_weight=1) = begin
    vals[1] = 2 - 16 * x[2] + 48 * x[1]^2
    vals[2] = -16 * x[1]
    vals[3] = 8.0
    vals .*= obj_weight
    return vals
  end
  h!(vals, x, y; obj_weight=1) = h!(vals, x; obj_weight=obj_weight)

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
  nlp = ADNLPModel!(
    f,
    x0,
    c!,
    [0.0],
    [0.0],
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
    (2 - 16 * x[2] + 48 * x[1]^2) * v[1] + (-16 * x[1]) * v[2];
    (-16 * x[1]) * v[1] + 8 * v[2]]
  @test hess(nlp, x) == [
    2 - 16 * x[2] + 48 * x[1]^2  0.0
    0.0     8.0]
  @test hprod(nlp, x, y, v) == hprod(nlp, x, y, v)
  @test hess(nlp, x, y) == hess(nlp, x, y)
  @test jprod(nlp, x, v) == [2]
  @test jtprod(nlp, x, y) == [y[1]; y[1]]
  @test jac(nlp, x) == [1 1]
  @test ghjvprod(nlp, x, g, v) == [0]
end
