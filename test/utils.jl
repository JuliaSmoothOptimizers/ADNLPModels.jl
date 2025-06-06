ReverseDiffAD(nvar, f) = ADNLPModels.ADModelBackend(
  nvar,
  f,
  gradient_backend = ADNLPModels.ReverseDiffADGradient,
  hprod_backend = ADNLPModels.ReverseDiffADHvprod,
  jprod_backend = ADNLPModels.ReverseDiffADJprod,
  jtprod_backend = ADNLPModels.ReverseDiffADJtprod,
  jacobian_backend = ADNLPModels.ReverseDiffADJacobian,
  hessian_backend = ADNLPModels.ReverseDiffADHessian,
)

function test_getter_setter(nlp)
  @test get_adbackend(nlp) == nlp.adbackend
  if typeof(nlp) <: ADNLPModel
    new_nlp = set_adbackend(nlp, ReverseDiffAD(nlp.meta.nvar, nlp.f))
  elseif typeof(nlp) <: ADNLSModel
    function F(x; nequ = nlp.nls_meta.nequ)
      Fx = similar(x, nequ)
      nlp.F!(Fx, x)
      return Fx
    end
    new_nlp = set_adbackend(nlp, ReverseDiffAD(nlp.meta.nvar, x -> sum(F(x) .^ 2)))
  end
  @test typeof(get_adbackend(new_nlp).gradient_backend) <: ADNLPModels.ReverseDiffADGradient
  @test typeof(get_adbackend(new_nlp).hprod_backend) <: ADNLPModels.ReverseDiffADHvprod
  @test typeof(get_adbackend(new_nlp).hessian_backend) <: ADNLPModels.ReverseDiffADHessian
  newer_nlp = set_adbackend(
    new_nlp,
    gradient_backend = ADNLPModels.ForwardDiffADGradient,
    jtprod_backend = ADNLPModels.GenericForwardDiffADJtprod(),
  )
  @test typeof(get_adbackend(newer_nlp).gradient_backend) <: ADNLPModels.ForwardDiffADGradient
  @test typeof(get_adbackend(newer_nlp).hprod_backend) <: ADNLPModels.ReverseDiffADHvprod
  @test typeof(get_adbackend(newer_nlp).jtprod_backend) <: ADNLPModels.GenericForwardDiffADJtprod
  @test typeof(get_adbackend(newer_nlp).hessian_backend) <: ADNLPModels.ReverseDiffADHessian
end

function test_allocations(nlp)
  x = nlp.meta.x0
  y = zeros(eltype(nlp.meta.x0), nlp.meta.ncon) 
  g = zeros(eltype(nlp.meta.x0), nlp.meta.nvar)
  @test_opt target_modules=(parentmodule(ADNLPModels.AbstractADNLPModel),) obj(nlp, x)  
  @test_opt target_modules=(parentmodule(ADNLPModels.AbstractADNLPModel),) cons!(nlp, x, y)
  @test_opt target_modules=(parentmodule(ADNLPModels.AbstractADNLPModel),) grad!(nlp, x, g)
end
