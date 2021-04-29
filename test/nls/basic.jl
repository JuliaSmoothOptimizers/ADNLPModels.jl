function autodiff_nls_test()
  for adbackend in (:ForwardDiffAD, :ZygoteAD, :ReverseDiffAD)
    @testset "autodiff_nls_test for $adbackend" begin
      F(x) = [x[1] - 1; x[2] - x[1]^2]
      nls = ADNLSModel(F, zeros(2), 2, adbackend = eval(adbackend)(F, zeros(2)))

      @test isapprox(residual(nls, ones(2)), zeros(2), rtol = 1e-8)
    end

    @testset "Constructors for ADNLSModel" begin
      F(x) = [x[1] - 1; x[2] - x[1]^2; x[1] * x[2]]
      x0 = ones(2)
      c(x) = [sum(x) - 1]
      lvar, uvar, lcon, ucon, y0 = -ones(2), ones(2), -ones(1), ones(1), zeros(1)
      badlvar, baduvar, badlcon, baducon, bady0 = -ones(3), ones(3), -ones(2), ones(2), zeros(2)
      unc_adbackend = eval(adbackend)(F, x0)
      con_adbackend = eval(adbackend)(F, c, x0, 1)
      nlp = ADNLSModel(F, x0, 3, adbackend = unc_adbackend)
      nlp = ADNLSModel(F, x0, 3, lvar, uvar, adbackend = unc_adbackend)
      nlp = ADNLSModel(F, x0, 3, c, lcon, ucon, adbackend = con_adbackend)
      nlp = ADNLSModel(F, x0, 3, c, lcon, ucon, y0 = y0, adbackend = con_adbackend)
      nlp = ADNLSModel(F, x0, 3, lvar, uvar, c, lcon, ucon, adbackend = con_adbackend)
      nlp = ADNLSModel(F, x0, 3, lvar, uvar, c, lcon, ucon, y0 = y0, adbackend = con_adbackend)
      @test_throws DimensionError ADNLSModel(F, x0, 3, badlvar, uvar, adbackend = unc_adbackend)
      @test_throws DimensionError ADNLSModel(F, x0, 3, lvar, baduvar, adbackend = unc_adbackend)
      @test_throws DimensionError ADNLSModel(F, x0, 3, c, badlcon, ucon, adbackend = con_adbackend)
      @test_throws DimensionError ADNLSModel(F, x0, 3, c, lcon, baducon, adbackend = con_adbackend)
      @test_throws DimensionError ADNLSModel(
        F,
        x0,
        3,
        c,
        lcon,
        ucon,
        y0 = bady0,
        adbackend = con_adbackend,
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
        adbackend = con_adbackend,
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
        adbackend = con_adbackend,
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
        adbackend = con_adbackend,
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
        adbackend = con_adbackend,
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
        adbackend = con_adbackend,
      )
    end
  end
end

autodiff_nls_test()
