mutable struct LinearRegression
  x::Vector
  y::Vector
end

function (regr::LinearRegression)(beta)
  r = regr.y .- beta[1] - beta[2] * regr.x
  return dot(r, r) / 2
end

function test_autodiff_model()
  for adbackend in (:ForwardDiffAD, :ReverseDiffAD)
    x0 = zeros(2)
    f(x) = dot(x, x)
    nlp = ADNLPModel(f, x0, adbackend = eval(adbackend)(length(x0), 0, f, x0))

    c(x) = [sum(x) - 1]
    nlp = ADNLPModel(f, x0, c, [0.0], [0.0], adbackend = eval(adbackend)(length(x0), 1, f, x0))
    @test obj(nlp, x0) == f(x0)

    x = range(-1, stop = 1, length = 100)
    y = 2x .+ 3 + randn(100) * 0.1
    regr = LinearRegression(x, y)
    nlp = ADNLPModel(regr, ones(2), adbackend = eval(adbackend)(2, 0, regr, ones(2)))
    β = [ones(100) x] \ y
    @test abs(obj(nlp, β) - norm(y .- β[1] - β[2] * x)^2 / 2) < 1e-12
    @test norm(grad(nlp, β)) < 1e-12

    @testset "Constructors for ADNLPModel" begin
      lvar, uvar, lcon, ucon, y0 = -ones(2), ones(2), -ones(1), ones(1), zeros(1)
      badlvar, baduvar, badlcon, baducon, bady0 = -ones(3), ones(3), -ones(2), ones(2), zeros(2)
      unc_adbackend = eval(adbackend)(2, f, x0)
      con_adbackend = eval(adbackend)(2, 1, f, x0)
      nlp = ADNLPModel(f, x0, adbackend = unc_adbackend)
      nlp = ADNLPModel(f, x0, lvar, uvar, adbackend = unc_adbackend)
      nlp = ADNLPModel(f, x0, c, lcon, ucon, adbackend = con_adbackend)
      nlp = ADNLPModel(f, x0, c, lcon, ucon, y0 = y0, adbackend = con_adbackend)
      nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon, adbackend = con_adbackend)
      nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon, y0 = y0, adbackend = con_adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, badlvar, uvar, adbackend = unc_adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, lvar, baduvar, adbackend = unc_adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, c, badlcon, ucon, adbackend = con_adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, c, lcon, baducon, adbackend = con_adbackend)
      @test_throws DimensionError ADNLPModel(
        f,
        x0,
        c,
        lcon,
        ucon,
        y0 = bady0,
        adbackend = con_adbackend,
      )
      @test_throws DimensionError ADNLPModel(
        f,
        x0,
        badlvar,
        uvar,
        c,
        lcon,
        ucon,
        adbackend = con_adbackend,
      )
      @test_throws DimensionError ADNLPModel(
        f,
        x0,
        lvar,
        baduvar,
        c,
        lcon,
        ucon,
        adbackend = con_adbackend,
      )
      @test_throws DimensionError ADNLPModel(
        f,
        x0,
        lvar,
        uvar,
        c,
        badlcon,
        ucon,
        adbackend = con_adbackend,
      )
      @test_throws DimensionError ADNLPModel(
        f,
        x0,
        lvar,
        uvar,
        c,
        lcon,
        baducon,
        adbackend = con_adbackend,
      )
      @test_throws DimensionError ADNLPModel(
        f,
        x0,
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

test_autodiff_model()
