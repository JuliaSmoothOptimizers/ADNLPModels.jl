function nlsmodelstest_autodiff(name; kwargs...)
  for problem in NLPModelsTest.nls_problems
    @testset "Checking NLPModelsTest tests on problem $problem" begin
      nls_from_T = eval(Meta.parse(lowercase(problem) * "_autodiff"))
      nls_ad = nls_from_T(; kwargs...)
      nls_man = eval(Meta.parse(problem))()

      nlss = AbstractNLSModel[nls_ad]
      # *_special problems are variant definitions of a model
      spc = "$(problem)_special"
      if isdefined(NLPModelsTest, Symbol(spc)) || isdefined(Main, Symbol(spc))
        push!(nlss, eval(Meta.parse(spc))())
      end

      exclude = if problem == "LLS"
        [hess_coord, hess]
      elseif problem == "MGH01"
        [hess_coord, hess, ghjvprod]
      else
        []
      end

      for nls in nlss
        show(IOBuffer(), nls)
      end

      @testset "Check Consistency" begin
        consistent_nlss([nlss; nls_man], exclude = exclude, linear_api = true)
      end
      @testset "Check dimensions" begin
        check_nls_dimensions.(nlss, exclude = exclude)
        check_nlp_dimensions.(nlss, exclude = exclude, linear_api = true)
      end
      @testset "Check multiple precision" begin
        for nls in nlss
          multiple_precision_nls(nls_from_T, exclude = exclude, linear_api = true)
        end
      end
      @testset "Check view subarray" begin
        view_subarray_nls.(nlss, exclude = exclude)
      end
    end
  end
end

nlsmodelstest_autodiff("OptimizedAD")
nlsmodelstest_autodiff(
  "ForwardDiff",
  gradient_backend = ADNLPModels.GenericForwardDiffADGradient,
  hprod_backend = ADNLPModels.GenericForwardDiffADHvprod,
  jprod_backend = ADNLPModels.GenericForwardDiffADJprod,
  jtprod_backend = ADNLPModels.ForwardDiffADJtprod,
  jacobian_backend = ADNLPModels.ForwardDiffADJacobian,
  hessian_backend = ADNLPModels.ForwardDiffADHessian,
  hprod_residual_backend = ADNLPModels.GenericForwardDiffADHvprod,
  jprod_residual_backend = ADNLPModels.GenericForwardDiffADJprod,
  jtprod_residual_backend = ADNLPModels.ForwardDiffADJtprod,
  jacobian_residual_backend = ADNLPModels.ForwardDiffADJacobian,
  hessian_residual_backend = ADNLPModels.ForwardDiffADHessian,
)
nlsmodelstest_autodiff(
  "ReverseDiff",
  gradient_backend = ADNLPModels.ReverseDiffADGradient,
  hprod_backend = ADNLPModels.ReverseDiffADHvprod,
  jprod_backend = ADNLPModels.ReverseDiffADJprod,
  jtprod_backend = ADNLPModels.ReverseDiffADJtprod,
  jacobian_backend = ADNLPModels.ReverseDiffADJacobian,
  hessian_backend = ADNLPModels.ReverseDiffADHessian,
  hprod_residual_backend = ADNLPModels.ReverseDiffADHvprod,
  jprod_residual_backend = ADNLPModels.GenericReverseDiffADJprod,
  jtprod_residual_backend = ADNLPModels.ReverseDiffADJtprod,
  jacobian_residual_backend = ADNLPModels.ReverseDiffADJacobian,
  hessian_residual_backend = ADNLPModels.ReverseDiffADHessian,
)
nlsmodelstest_autodiff(
  "Zygote",
  gradient_backend = ADNLPModels.ZygoteADGradient,
  jprod_backend = ADNLPModels.ZygoteADJprod,
  jtprod_backend = ADNLPModels.ZygoteADJtprod,
  jacobian_backend = ADNLPModels.ZygoteADJacobian,
  hessian_backend = ADNLPModels.ZygoteADHessian,
  jprod_residual_backend = ADNLPModels.ZygoteADJprod,
  jtprod_residual_backend = ADNLPModels.ZygoteADJtprod,
  jacobian_residual_backend = ADNLPModels.ZygoteADJacobian,
  hessian_residual_backend = ADNLPModels.ZygoteADHessian,
)
