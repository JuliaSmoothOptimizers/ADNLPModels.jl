function autodiff_nls_test()
  for adbackend in (ForwardDiffAD, ZygoteAD, ReverseDiffAD)
    @testset "autodiff_nls_test for $adbackend" begin
      F(x) = [x[1] - 1; x[2] - x[1]^2]
      nls = ADNLSModel(F, zeros(2), 2, backend = adbackend)

      @test isapprox(residual(nls, ones(2)), zeros(2), rtol = 1e-8)
    end

    @testset "Constructors for ADNLSModel" begin
      F(x) = [x[1] - 1; x[2] - x[1]^2; x[1] * x[2]]
      x0 = ones(2)
      c(x) = [sum(x) - 1]
      lvar, uvar, lcon, ucon, y0 = -ones(2), ones(2), -ones(1), ones(1), zeros(1)
      badlvar, baduvar, badlcon, baducon, bady0 = -ones(3), ones(3), -ones(2), ones(2), zeros(2)
      nlp = ADNLSModel(F, x0, 3, backend = adbackend)
      nlp = ADNLSModel(F, x0, 3, lvar, uvar, backend = adbackend)
      nlp = ADNLSModel(F, x0, 3, c, lcon, ucon, backend = adbackend)
      nlp = ADNLSModel(F, x0, 3, c, lcon, ucon, y0 = y0, backend = adbackend)
      nlp = ADNLSModel(F, x0, 3, lvar, uvar, c, lcon, ucon, backend = adbackend)
      nlp = ADNLSModel(F, x0, 3, lvar, uvar, c, lcon, ucon, y0 = y0, backend = adbackend)
      @test_throws DimensionError ADNLSModel(F, x0, 3, badlvar, uvar, backend = adbackend)
      @test_throws DimensionError ADNLSModel(F, x0, 3, lvar, baduvar, backend = adbackend)
      @test_throws DimensionError ADNLSModel(F, x0, 3, c, badlcon, ucon, backend = adbackend)
      @test_throws DimensionError ADNLSModel(F, x0, 3, c, lcon, baducon, backend = adbackend)
      @test_throws DimensionError ADNLSModel(
        F,
        x0,
        3,
        c,
        lcon,
        ucon,
        y0 = bady0,
        backend = adbackend,
      )
      @test_throws DimensionError ADNLSModel(
        F,
        x0,
        3,
        badlvar,
        uvar,
        c,
        lcon,
        ucon,
        backend = adbackend,
      )
      @test_throws DimensionError ADNLSModel(
        F,
        x0,
        3,
        lvar,
        baduvar,
        c,
        lcon,
        ucon,
        backend = adbackend,
      )
      @test_throws DimensionError ADNLSModel(
        F,
        x0,
        3,
        lvar,
        uvar,
        c,
        badlcon,
        ucon,
        backend = adbackend,
      )
      @test_throws DimensionError ADNLSModel(
        F,
        x0,
        3,
        lvar,
        uvar,
        c,
        lcon,
        baducon,
        backend = adbackend,
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
        y0 = bady0,
        backend = adbackend,
      )
    end
  end
end

autodiff_nls_test()
