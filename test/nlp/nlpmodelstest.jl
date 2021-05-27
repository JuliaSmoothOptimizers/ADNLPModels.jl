
@testset "AD backend - $(adbackend)" for adbackend in backends()
  for problem in NLPModelsTest.nlp_problems
    @testset "Checking NLPModelsTest tests on problem $problem" begin
      nlp_ad = eval(Meta.parse(lowercase(problem) * "_autodiff"))()
      nlp_ad.adbackend = eval(adbackend)(length(nlp_ad.meta.x0))
      nlp_man = eval(Meta.parse(problem))()

      show(IOBuffer(), nlp_ad)

      nlps = [nlp_ad, nlp_man]
      @testset "Check Consistency" begin
        consistent_nlps(nlps, exclude = [])
      end
      @testset "Check dimensions" begin
        check_nlp_dimensions(nlp_ad, exclude = [])
      end
      @testset "Check multiple precision" begin
        multiple_precision_nlp(nlp_ad, exclude = [])
      end
    end
  end
end
