function autodiff_nls_test(name; kwargs...)
  @testset "autodiff_nls_test for $name" begin
    F(x) = [x[1] - 1; x[2] - x[1]^2]
    nls = ADNLSModel(F, zeros(2), 2; kwargs...)

    @test isapprox(residual(nls, ones(2)), zeros(2), rtol = 1e-8)
  end

  @testset "Constructors for ADNLSModel" begin
    F(x) = [x[1] - 1; x[2] - x[1]^2; x[1] * x[2]]
    x0 = ones(2)
    c(x) = [sum(x) - 1]
    lvar, uvar, lcon, ucon, y0 = -ones(2), ones(2), -ones(1), ones(1), zeros(1)
    badlvar, baduvar, badlcon, baducon, bady0 = -ones(3), ones(3), -ones(2), ones(2), zeros(2)
    nlp = ADNLSModel(F, x0, 3; kwargs...)
    nlp = ADNLSModel(F, x0, 3, lvar, uvar; kwargs...)
    nlp = ADNLSModel(F, x0, 3, c, lcon, ucon; kwargs...)
    nlp = ADNLSModel(F, x0, 3, c, lcon, ucon, y0 = y0; kwargs...)
    nlp = ADNLSModel(F, x0, 3, lvar, uvar, c, lcon, ucon; kwargs...)
    nlp = ADNLSModel(F, x0, 3, lvar, uvar, c, lcon, ucon, y0 = y0; kwargs...)
    @test_throws DimensionError ADNLSModel(F, x0, 3, badlvar, uvar; kwargs...)
    @test_throws DimensionError ADNLSModel(F, x0, 3, lvar, baduvar; kwargs...)
    @test_throws DimensionError ADNLSModel(F, x0, 3, c, badlcon, ucon; kwargs...)
    @test_throws DimensionError ADNLSModel(F, x0, 3, c, lcon, baducon; kwargs...)
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      c,
      lcon,
      ucon,
      y0 = bady0;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      badlvar,
      uvar,
      c,
      lcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      lvar,
      baduvar,
      c,
      lcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      lvar,
      uvar,
      c,
      badlcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      lvar,
      uvar,
      c,
      lcon,
      baducon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      lvar,
      uvar,
      c,
      lcon,
      ucon,
      y0 = bady0;
      kwargs...,
    )
  end
end

autodiff_nls_test("ForwardDiff")
autodiff_nls_test(
  "ReverseDiff",
  gradient_backend = ADNLPModels.ReverseDiffADGradient,
  hprod_backend = ADNLPModels.ReverseDiffADHvprod,
  jprod_backend = ADNLPModels.ReverseDiffADJprod,
  jtprod_backend = ADNLPModels.ReverseDiffADJtprod,
  jacobian_backend = ADNLPModels.ReverseDiffADJacobian,
  hessian_backend = ADNLPModels.ReverseDiffADHessian,
)
autodiff_nls_test(
  "Zygote",
  gradient_backend = ADNLPModels.ZygoteADGradient,
  jprod_backend = ADNLPModels.ZygoteADJprod,
  jtprod_backend = ADNLPModels.ZygoteADJtprod,
  jacobian_backend = ADNLPModels.ZygoteADJacobian,
  hessian_backend = ADNLPModels.ZygoteADHessian,
)

