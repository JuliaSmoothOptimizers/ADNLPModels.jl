# script that tests ADNLPModels over OptimizationProblems.jl problems

# AD deps
using ForwardDiff, ReverseDiff

# JSO packages
using ADNLPModels, OptimizationProblems, NLPModels, Test

# Comparison with JuMP
using JuMP, NLPModelsJuMP

names = OptimizationProblems.meta[!, :name]

for pb in names
  @info pb

  nlp = try
    OptimizationProblems.ADNLPProblems.eval(Meta.parse(pb))(backend = :default, show_time = true)
  catch e
    println("Error $e with ADNLPModel")
    continue
  end

  jum = try
    MathOptNLPModel(OptimizationProblems.PureJuMP.eval(Meta.parse(pb))())
  catch e
    println("Error $e with JuMP")
    continue
  end

  n, m = nlp.meta.nvar, nlp.meta.ncon
  x = 10 * [-(-1.0)^i for i = 1:n] # find a better point in the domain.
  v = 10 * [-(-1.0)^i for i = 1:n]
  y = 3.14 * ones(m)

  # test the main functions in the API
  try
    @testset "Test NLPModel API $(nlp.meta.name)" begin
      @test grad(nlp, x) ≈ grad(jum, x)
      @test hess(nlp, x) ≈ hess(jum, x)
      @test hess(nlp, x, y) ≈ hess(jum, x, y)
      @test hprod(nlp, x, v) ≈ hprod(jum, x, v)
      @test hprod(nlp, x, y, v) ≈ hprod(jum, x, y, v)
      if nlp.meta.ncon > 0
        @test jac(nlp, x) ≈ jac(jum, x)
        @test jprod(nlp, x, v) ≈ jprod(jum, x, v)
        @test jtprod(nlp, x, y) ≈ jtprod(jum, x, y)
      end
    end
  catch e
    println("Error $e with API")
    continue
  end
end
