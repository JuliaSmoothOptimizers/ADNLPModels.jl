using ADNLPModels, LinearAlgebra, NLPModels, NLPModelsModifiers, NLPModelsTest, SparseArrays, Test

for problem in NLPModelsTest.nlp_problems ∪ ["GENROSE"]
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in NLPModelsTest.nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end
for problem in setdiff(NLPModelsTest.nlp_problems)
  include("radnlp/problems/$(lowercase(problem)).jl")
end

#=
include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")
=#
include("radnlp/basic.jl") #Doesn't work
include("radnlp/nlpmodelstest.jl") #Partially work

#=
#List of problems used in extansive tests
list_problems = ["arglina", "arglinb", "arglinc", "arwhead", "bdqrtic", "beale", "broydn7d",
             "brybnd", "chainwoo", "chnrosnb", "cosine", "cragglvy", "dixon3dq", "dqdrtic",
             "dqrtic", "edensch", "eg2", "engval1", "errinros", "extrosnb", "fletcbv2",
             "fletcbv3", "fletchcr", "freuroth", "genhumps", "genrose", "genrose_nash",
             "indef", "liarwhd", "morebv", "ncb20", "ncb20b", "noncvxu2", "noncvxun",
             "nondia", "nondquar", "nzf1", "penalty2", "penalty3", "powellsg", "power",
             "quartc", "sbrybnd", "schmvett", "scosine", "sparsine", "sparsqur", "srosenbr",
             "sinquad", "tointgss", "tquartic", "tridia", "vardim", "woods"]
for pb in list_problems
    include("problems/$(lowercase(pb)).jl")
end

#=
# List of problems whose symbolic hessian fails:
# cf. issue posted on Symbolics.jl: https://github.com/JuliaSymbolics/Symbolics.jl/issues/108
"brownden", "arwhead", "bdqrtic", "beale", "broydn7d", "brybnd", "cragglvy", "eg2", "freuroth", "genhumps"
"indef", "ncb20", "ncb20b", "noncvxun", "nondquar", "nzf1", "penalty2", "penalty3","powellsg", "power",
"sbrybnd", "schmvett", "sparsine", "sparsqur", "sinquad", "tquartic", "vardim"
works but super slow:
"arglinb", "arglinc"
=#

for problem in list_problems
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
=#
