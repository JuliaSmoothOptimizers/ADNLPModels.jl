@testset "AD backend - $(adbackend)" for adbackend in (:ForwardDiffAD, :ZygoteAD, :ReverseDiffAD)
  for problem in NLPModelsTest.nls_problems
    @testset "Checking NLPModelsTest tests on problem $problem" begin
      nls_ad = eval(Meta.parse(lowercase(problem) * "_autodiff"))()
      nls_ad.adbackend = eval(adbackend)(nls_ad.F, nls_ad.meta.x0)
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
        consistent_nlss([nlss; nls_man], exclude = exclude)
      end
      @testset "Check dimensions" begin
        check_nls_dimensions.(nlss, exclude = exclude)
        check_nlp_dimensions.(nlss, exclude = exclude)
      end
      @testset "Check multiple precision" begin
        for nls in nlss
          multiple_precision_nls(nls, exclude = exclude)
        end
      end
      @testset "Check view subarray" begin
        view_subarray_nls.(nlss, exclude = exclude)
      end
    end
  end
end
