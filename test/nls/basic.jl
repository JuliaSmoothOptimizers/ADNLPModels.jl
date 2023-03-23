function autodiff_nls_test(name; kwargs...)
  @testset "autodiff_nls_test for $name" begin
    F(x) = [x[1] - 1; x[2] - x[1]^2]
    nls = ADNLSModel(F, zeros(2), 2; kwargs...)

    @test isapprox(residual(nls, ones(2)), zeros(2), rtol = 1e-8)

    test_getter_setter(nls)
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
    @test_throws DimensionError ADNLSModel(F, x0, 3, c, lcon, ucon, y0 = bady0; kwargs...)
    @test_throws DimensionError ADNLSModel(F, x0, 3, badlvar, uvar, c, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLSModel(F, x0, 3, lvar, baduvar, c, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLSModel(F, x0, 3, lvar, uvar, c, badlcon, ucon; kwargs...)
    @test_throws DimensionError ADNLSModel(F, x0, 3, lvar, uvar, c, lcon, baducon; kwargs...)
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

    clinrows, clincols, clinvals = ones(Int, 2), ones(Int, 2), ones(2)
    badclinrows, badclincols, badclinvals = ones(Int, 3), ones(Int, 3), ones(3)
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      clinrows,
      clincols,
      clinvals,
      badlcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      clinrows,
      clincols,
      clinvals,
      lcon,
      baducon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      badclinrows,
      clincols,
      clinvals,
      lcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      clinrows,
      badclincols,
      clinvals,
      lcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      clinrows,
      clincols,
      badclinvals,
      lcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      clinrows,
      clincols,
      clinvals,
      c,
      badlcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      clinrows,
      clincols,
      clinvals,
      c,
      lcon,
      baducon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      badclinrows,
      clincols,
      clinvals,
      c,
      lcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      clinrows,
      badclincols,
      clinvals,
      c,
      lcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      clinrows,
      clincols,
      badclinvals,
      c,
      lcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      badlvar,
      uvar,
      clinrows,
      clincols,
      clinvals,
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
      clinrows,
      clincols,
      clinvals,
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
      clinrows,
      clincols,
      clinvals,
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
      clinrows,
      clincols,
      clinvals,
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
      badclinrows,
      clincols,
      clinvals,
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
      clinrows,
      badclincols,
      clinvals,
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
      clinrows,
      clincols,
      badclinvals,
      lcon,
      ucon;
      kwargs...,
    )
    @test_throws DimensionError ADNLSModel(
      F,
      x0,
      3,
      badlvar,
      uvar,
      clinrows,
      clincols,
      clinvals,
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
      clinrows,
      clincols,
      clinvals,
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
      clinrows,
      clincols,
      clinvals,
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
      clinrows,
      clincols,
      clinvals,
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
      badclinrows,
      clincols,
      clinvals,
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
      clinrows,
      badclincols,
      clinvals,
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
      clinrows,
      clincols,
      badclinvals,
      c,
      lcon,
      ucon;
      kwargs...,
    )

    A = sparse(clinrows, clincols, clinvals)
    nls = ADNLSModel(F, x0, 3, A, c, -ones(2), ones(2))
    @test A == sparse(nls.clinrows, nls.clincols, nls.clinvals)
    nls = ADNLSModel(F, x0, 3, A, lcon, ucon)
    @test A == sparse(nls.clinrows, nls.clincols, nls.clinvals)
    nls = ADNLSModel(F, x0, 3, lvar, uvar, A, c, -ones(2), ones(2))
    @test A == sparse(nls.clinrows, nls.clincols, nls.clinvals)
    nls = ADNLSModel(F, x0, 3, lvar, uvar, A, lcon, ucon)
    @test A == sparse(nls.clinrows, nls.clincols, nls.clinvals)
  end
end

autodiff_nls_test("OptimizedAD")
autodiff_nls_test(
  "ForwardDiff",
  gradient_backend = ADNLPModels.GenericForwardDiffADGradient,
  hprod_backend = ADNLPModels.ForwardDiffADHvprod,
  jprod_backend = ADNLPModels.GenericForwardDiffADJprod,
  jtprod_backend = ADNLPModels.ForwardDiffADJtprod,
  jacobian_backend = ADNLPModels.ForwardDiffADJacobian,
  hessian_backend = ADNLPModels.ForwardDiffADHessian,
)
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
