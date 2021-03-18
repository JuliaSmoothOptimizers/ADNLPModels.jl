for problem in setdiff(NLPModelsTest.nlp_problems,["BROWNDEN"])
  @testset "Checking NLPModelsTest tests on problem $problem" begin
    nlp_ad = eval(Meta.parse(lowercase(problem) * "_autodiff"))()
    nlp_rad = eval(Meta.parse(lowercase(problem) * "_radnlp"))()
    nlp_man = eval(Meta.parse(problem))()
    
    show(IOBuffer(), nlp_rad)
    
    nlps = [nlp_ad, nlp_man, nlp_rad]
    @testset "Check Consistency" begin
      #consistent_nlps(nlps)
    end
    @testset "Check dimensions" begin
      check_nlp_dimensions(nlp_rad)
    end
    @testset "Check multiple precision" begin
      @info "TODOs"
      #multiple_precision_nlp(nlp_ad)
    end
    @testset "Check view subarray" begin
      @info "TODOs"
      #view_subarray_nlp(nlp_ad)
    end
    @testset "Check coordinate memory" begin
      @info "TODOs"
      #coord_memory_nlp(nlp_ad)
    end

    @testset "Extra consistency" begin
      pb_radnlp = eval(Meta.parse("$(lowercase(problem))_radnlp()"))
      pb_adnlp = eval(Meta.parse("$(lowercase(problem))_autodiff()"))
    
      @test pb_radnlp.meta.nvar == pb_adnlp.meta.nvar
    
      x = rand(pb_radnlp.meta.nvar)
      @test obj(pb_radnlp, x) ≈ obj(pb_adnlp, x)
      @test grad(pb_radnlp, x) ≈ grad(pb_adnlp, x)
      @test hess(pb_radnlp, x) ≈ hess(pb_adnlp, x)
    
      v = rand(pb_radnlp.meta.nvar)
      #@test hprod(pb_radnlp, x, v) ≈ hprod(pb_adnlp, x, v)
    
      @test pb_radnlp.meta.ncon == pb_adnlp.meta.ncon
      if pb_radnlp.meta.ncon > 0
        @test cons(pb_radnlp, x) ≈ cons(pb_adnlp, x)
        @test jac(pb_radnlp, x)  ≈ jac(pb_adnlp, x)
      end
    end
  end
end
