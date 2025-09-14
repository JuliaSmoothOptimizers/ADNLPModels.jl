function test_allocations(nlp::ADNLPModel)
  x = nlp.meta.x0
  y = zeros(eltype(nlp.meta.x0), nlp.meta.ncon) 
  g = zeros(eltype(nlp.meta.x0), nlp.meta.nvar)
  @test_opt target_modules=(ADNLPModels,) obj(nlp, x)  
  @test_opt target_modules=(ADNLPModels,) cons!(nlp, x, y)
  @test_opt target_modules=(ADNLPModels,) grad!(nlp, x, g)
end

function test_allocations(nlp::ADNLSModel)
  x = nlp.meta.x0
  y = zeros(eltype(nlp.meta.x0), nlp.meta.ncon) 
  g = zeros(eltype(nlp.meta.x0), nlp.meta.nvar)
  Fx = zeros(eltype(nlp.meta.x0), nlp.nls_meta.nequ)
  @test_opt target_modules=(ADNLPModels,) ignored_modules=(ForwardDiff,) obj(nlp, x)  
  @test_opt target_modules=(ADNLPModels,) ignored_modules=(ForwardDiff,) cons!(nlp, x, y)
  @test_opt target_modules=(ADNLPModels,) ignored_modules=(ForwardDiff,) grad!(nlp, x, g, Fx)
  @test_opt target_modules=(ADNLPModels,) ignored_modules=(ForwardDiff,) residual!(nlp, x, Fx)
end
