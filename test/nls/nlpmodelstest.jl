@testset "Checking NLPModelsTest (NLS) tests with $backend" for backend in
                                                                keys(ADNLPModels.predefined_backend)
  @testset "Checking NLPModelsTest tests on problem $problem" for problem in
                                                                  NLPModelsTest.nls_problems
    nls_from_T = eval(Meta.parse(lowercase(problem) * "_autodiff"))
    nls_ad = nls_from_T(; backend = backend)
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
      multiple_precision_nls(nls_from_T, exclude = exclude, linear_api = true)
    end
    if backend != :enzyme
      @testset "Check view subarray" begin
        view_subarray_nls.(nlss, exclude = exclude)
      end
    end
  end
end
