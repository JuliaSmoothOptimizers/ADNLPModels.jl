mutable struct LinearRegression
  x::Vector
  y::Vector
end

function (regr::LinearRegression)(beta)
  r = regr.y .- beta[1] - beta[2] * regr.x
  return dot(r, r) / 2
end

ReverseDiffAD() = ADNLPModels.ADModelBackend(
  ADNLPModels.ReverseDiffADGradient(nothing),
  ADNLPModels.ReverseDiffADHvprod(),
  ADNLPModels.ReverseDiffADJprod(),
  ADNLPModels.ReverseDiffADJtprod(),
  ADNLPModels.ReverseDiffADJacobian(0),
  ADNLPModels.ReverseDiffADHessian(0),
  ADNLPModels.ForwardDiffADGHjvprod(),
)
ZygoteAD() = ADNLPModels.ADModelBackend(
  ADNLPModels.ZygoteADGradient(),
  ADNLPModels.ForwardDiffADHvprod(),
  ADNLPModels.ZygoteADJprod(),
  ADNLPModels.ZygoteADJtprod(),
  ADNLPModels.ZygoteADJacobian(0),
  ADNLPModels.ZygoteADHessian(0),
  ADNLPModels.ForwardDiffADGHjvprod(),
)

function test_autodiff_backend_error()
  @testset "Error without loading package - $backend" for backend in [:ZygoteAD]
    adbackend = eval(backend)()
    @test_throws ArgumentError gradient(adbackend.gradient_backend, sum, [1.0])
    @test_throws ArgumentError gradient!(adbackend.gradient_backend, [1.0], sum, [1.0])
    @test_throws ArgumentError jacobian(adbackend.jacobian_backend, identity, [1.0])
    @test_throws ArgumentError hessian(adbackend.hessian_backend, sum, [1.0])
    @test_throws ArgumentError Jprod(adbackend.jprod_backend, [1.0], identity, [1.0])
    @test_throws ArgumentError Jtprod(adbackend.jtprod_backend, [1.0], identity, [1.0])
  end
end

function test_autodiff_model(name; kwargs...)
  x0 = zeros(2)
  f(x) = dot(x, x)
  nlp = ADNLPModel(f, x0; kwargs...)

  c(x) = [sum(x) - 1]
  nlp = ADNLPModel(f, x0, c, [0.0], [0.0]; kwargs...)
  @test obj(nlp, x0) == f(x0)

  x = range(-1, stop = 1, length = 100)
  y = 2x .+ 3 + randn(100) * 0.1
  regr = LinearRegression(x, y)
  nlp = ADNLPModel(regr, ones(2); kwargs...)
  β = [ones(100) x] \ y
  @test abs(obj(nlp, β) - norm(y .- β[1] - β[2] * x)^2 / 2) < 1e-12
  @test norm(grad(nlp, β)) < 1e-12
  
  test_getter_setter(nlp)

  @testset "Constructors for ADNLPModel with $name" begin
    lvar, uvar, lcon, ucon, y0 = -ones(2), ones(2), -ones(1), ones(1), zeros(1)
    badlvar, baduvar, badlcon, baducon, bady0 = -ones(3), ones(3), -ones(2), ones(2), zeros(2)
    nlp = ADNLPModel(f, x0; kwargs...)
    nlp = ADNLPModel(f, x0, lvar, uvar; kwargs...)
    nlp = ADNLPModel(f, x0, c, lcon, ucon; kwargs...)
    nlp = ADNLPModel(f, x0, c, lcon, ucon, y0 = y0; kwargs...)
    nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon; kwargs...)
    nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon, y0 = y0; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, badlvar, uvar; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, baduvar; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, c, badlcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, c, lcon, baducon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, c, lcon, ucon, y0 = bady0; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, badlvar, uvar, c, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, baduvar, c, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, c, badlcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, c, lcon, baducon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon; y0 = bady0, kwargs...)

    clinrows, clincols, clinvals = ones(Int, 2), ones(Int, 2), ones(2)
    badclinrows, badclincols, badclinvals = ones(Int, 3), ones(Int, 3), ones(3)
    @test_throws DimensionError ADNLPModel(f, x0, clinrows, clincols, clinvals, badlcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, clinrows, clincols, clinvals, lcon, baducon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, badclinrows, clincols, clinvals, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, clinrows, badclincols, clinvals, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, clinrows, clincols, badclinvals, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, clinrows, clincols, clinvals, c, badlcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, clinrows, clincols, clinvals, c, lcon, baducon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, badclinrows, clincols, clinvals, c, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, clinrows, badclincols, clinvals, c, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, clinrows, clincols, badclinvals, c, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, badlvar, uvar, clinrows, clincols, clinvals, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, baduvar, clinrows, clincols, clinvals, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, clinrows, clincols, clinvals, badlcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, clinrows, clincols, clinvals, lcon, baducon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, badclinrows, clincols, clinvals, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, clinrows, badclincols, clinvals, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, clinrows, clincols, badclinvals, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, badlvar, uvar, clinrows, clincols, clinvals, c, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, baduvar, clinrows, clincols, clinvals, c, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, clinrows, clincols, clinvals, c, badlcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, clinrows, clincols, clinvals, c, lcon, baducon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, badclinrows, clincols, clinvals, c, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, clinrows, badclincols, clinvals, c, lcon, ucon; kwargs...)
    @test_throws DimensionError ADNLPModel(f, x0, lvar, uvar, clinrows, clincols, badclinvals, c, lcon, ucon; kwargs...)

    A = sparse(clinrows, clincols, clinvals)
    nlp = ADNLPModel(f, x0, A, c, lcon, ucon)
    @test A == sparse(nlp.clinrows, nlp.clincols, nlp.clinvals)
    nlp = ADNLPModel(f, x0, A, lcon, ucon)
    @test A == sparse(nlp.clinrows, nlp.clincols, nlp.clinvals)
    nlp = ADNLPModel(f, x0, lvar, uvar, A, c, lcon, ucon)
    @test A == sparse(nlp.clinrows, nlp.clincols, nlp.clinvals)
    nlp = ADNLPModel(f, x0, lvar, uvar, A, lcon, ucon)
    @test A == sparse(nlp.clinrows, nlp.clincols, nlp.clinvals)
    nlp = ADNLPModel(f, x0, lvar, uvar, A, lcon, ucon)
    @test A == sparse(nlp.clinrows, nlp.clincols, nlp.clinvals)
  end
end

# Test the argument error without loading the packages
test_autodiff_backend_error()

# Automatically loads the code for Zygote with Requires
import Zygote

test_autodiff_model("ForwardDiff")
test_autodiff_model(
  "ReverseDiff",
  gradient_backend = ADNLPModels.ReverseDiffADGradient,
  hprod_backend = ADNLPModels.ReverseDiffADHvprod,
  jprod_backend = ADNLPModels.ReverseDiffADJprod,
  jtprod_backend = ADNLPModels.ReverseDiffADJtprod,
  jacobian_backend = ADNLPModels.ReverseDiffADJacobian,
  hessian_backend = ADNLPModels.ReverseDiffADHessian,
)
test_autodiff_model(
  "Zygote",
  gradient_backend = ADNLPModels.ZygoteADGradient,
  jprod_backend = ADNLPModels.ZygoteADJprod,
  jtprod_backend = ADNLPModels.ZygoteADJtprod,
  jacobian_backend = ADNLPModels.ZygoteADJacobian,
  hessian_backend = ADNLPModels.ZygoteADHessian,
)
