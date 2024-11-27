@testset "Checking NLPModelsTest (NLP) tests with $backend" for backend in
                                                                keys(ADNLPModels.predefined_backend)
  @testset "Checking NLPModelsTest tests on problem $problem" for problem in
                                                                  NLPModelsTest.nlp_problems
    nlp_from_T = eval(Meta.parse(lowercase(problem) * "_autodiff"))
    nlp_ad = nlp_from_T(; backend = backend)
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
