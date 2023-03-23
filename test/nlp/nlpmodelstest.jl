
function nlpmodelstest_autodiff(name; kwargs...)
  for problem in NLPModelsTest.nlp_problems
    @testset "Checking NLPModelsTest tests on problem $problem with $name" begin
      nlp_from_T = eval(Meta.parse(lowercase(problem) * "_autodiff"))
      nlp_ad = nlp_from_T(; kwargs...)
      nlp_man = eval(Meta.parse(problem))()

      show(IOBuffer(), nlp_ad)

      nlps = [nlp_ad, nlp_man]
      @testset "Check Consistency" begin
        consistent_nlps(nlps, exclude = [], linear_api = true, reimplemented = ["jtprod"])
      end
      @testset "Check dimensions" begin
        check_nlp_dimensions(nlp_ad, exclude = [], linear_api = true)
      end
      @testset "Check multiple precision" begin
        multiple_precision_nlp(nlp_from_T, exclude = [], linear_api = true)
      end
      @testset "Check view subarray" begin
        view_subarray_nlp(nlp_ad, exclude = [])
      end
      @testset "Check coordinate memory" begin
        coord_memory_nlp(nlp_ad, exclude = [], linear_api = true)
      end
    end
  end
end

nlpmodelstest_autodiff("OptimizedAD")
nlpmodelstest_autodiff(
  "ForwardDiff",
  gradient_backend = ADNLPModels.GenericForwardDiffADGradient,
  hprod_backend = ADNLPModels.GenericForwardDiffADHvprod,
  jprod_backend = ADNLPModels.GenericForwardDiffADJprod,
  jtprod_backend = ADNLPModels.ForwardDiffADJtprod,
  jacobian_backend = ADNLPModels.ForwardDiffADJacobian,
  hessian_backend = ADNLPModels.ForwardDiffADHessian,
)
nlpmodelstest_autodiff(
  "ReverseDiff",
  gradient_backend = ADNLPModels.ReverseDiffADGradient,
  hprod_backend = ADNLPModels.ReverseDiffADHvprod,
  jprod_backend = ADNLPModels.ReverseDiffADJprod,
  jtprod_backend = ADNLPModels.GenericReverseDiffADJtprod,
  jacobian_backend = ADNLPModels.ReverseDiffADJacobian,
  hessian_backend = ADNLPModels.ReverseDiffADHessian,
)
nlpmodelstest_autodiff(
  "Zygote",
  gradient_backend = ADNLPModels.ZygoteADGradient,
  jprod_backend = ADNLPModels.ZygoteADJprod,
  jtprod_backend = ADNLPModels.ZygoteADJtprod,
  jacobian_backend = ADNLPModels.ZygoteADJacobian,
  hessian_backend = ADNLPModels.ZygoteADHessian,
)
