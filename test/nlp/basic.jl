mutable struct LinearRegression
  x :: Vector
  y :: Vector
end

function (regr::LinearRegression)(beta)
  r = regr.y .- beta[1] - beta[2] * regr.x
  return dot(r, r) / 2
end

function test_autodiff_backend_error()
  @testset "Error without loading package - $adbackend" for adbackend in [
    ZygoteAD(), ReverseDiffAD(),
  ]
    @test_throws ArgumentError gradient(adbackend, sum, [1.0])
    @test_throws ArgumentError gradient!(adbackend, [1.0], sum, [1.0])
    @test_throws ArgumentError jacobian(adbackend, identity, [1.0])
    @test_throws ArgumentError hessian(adbackend, sum, [1.0])
    @test_throws ArgumentError Jprod(adbackend, [1.0], identity, [1.0])
    @test_throws ArgumentError Jtprod(adbackend, [1.0], identity, [1.0])
    @test_throws ArgumentError Hvprod(adbackend, sum, [1.0], [1.0])
  end
end

function test_autodiff_model()
  for adbackend in [
    ForwardDiffAD(), ZygoteAD(), ReverseDiffAD(),
  ]
    x0 = zeros(2)
    f(x) = dot(x,x)
    nlp = ADNLPModel(f, x0, adbackend = adbackend)

    c(x) = [sum(x) - 1]
    nlp = ADNLPModel(f, x0, c, [0], [0], adbackend = adbackend)
    @test obj(nlp, x0) == f(x0)

    x = range(-1, stop=1, length=100)
    y = 2x .+ 3 + randn(100) * 0.1
    regr = LinearRegression(x, y)
    nlp = ADNLPModel(regr, ones(2), adbackend = adbackend)
    β = [ones(100) x] \ y
    @test abs(obj(nlp, β) - norm(y .- β[1] - β[2] * x)^2 / 2) < 1e-12
    @test norm(grad(nlp, β)) < 1e-12

    @testset "Constructors for ADNLPModel" begin
      lvar, uvar, lcon, ucon, y0 = -ones(2), ones(2), -ones(1), ones(1), zeros(1)
      badlvar, baduvar, badlcon, baducon, bady0 = -ones(3), ones(3), -ones(2), ones(2), zeros(2)
      nlp = ADNLPModel(f, x0, adbackend = adbackend)
      nlp = ADNLPModel(f, x0, lvar, uvar, adbackend = adbackend)
      nlp = ADNLPModel(f, x0, c, lcon, ucon, adbackend = adbackend)
      nlp = ADNLPModel(f, x0, c, lcon, ucon, y0=y0, adbackend = adbackend)
      nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon, adbackend = adbackend)
      nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon, y0=y0, adbackend = adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, badlvar, uvar, adbackend = adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, lvar, baduvar, adbackend = adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, c, badlcon, ucon, adbackend = adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, c, lcon, baducon, adbackend = adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, c, lcon, ucon, y0=bady0, adbackend = adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, badlvar, uvar, c, lcon, ucon, adbackend = adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, lvar, baduvar, c, lcon, ucon, adbackend = adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, c, badlcon, ucon, adbackend = adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, c, lcon, baducon, adbackend = adbackend)
      @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon, y0=bady0, adbackend = adbackend)
    end
  end
end

# Test the argument error without loading the packages
test_autodiff_backend_error()

# Automatically loads the code for Zygote and ReverseDiff with Requires
import Zygote, ReverseDiff

test_autodiff_model()
