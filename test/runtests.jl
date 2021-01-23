using LinearAlgebra, NLPModels, Test
#This package
using ADNLPModels

#Lib of test functions
#I. Classical tests from NLPModels
include("check_dimensions.jl") #copy-paste from NLPModels #check_nlp_dimensions(nlp; exclude_hess=false)
include("consistency.jl") #copy-paste from NLPModels 
include("multiple-precision.jl") #copy-paste from NLPModels 
include("test_autodiff_model.jl") #test_autodiff_model()

#Tests 0-problems
#test_autodiff_model() #uses constraints

#Problems from NLPModels
problems = ["hs5", "brownden"]
problems2 = ["arglina", "arglinb", "arglinc", "arwhead", "bdqrtic", "beale", "broydn7d",
             "brybnd", "chainwoo", "chnrosnb", "cosine", "cragglvy", "dixon3dq", "dqdrtic",
             "dqrtic", "edensch", "eg2", "engval1", "errinros", "extrosnb", "fletcbv2",
             "fletcbv3", "fletchcr", "freuroth", "genhumps", "genrose", "genrose_nash",
             "indef", "liarwhd", "morebv", "ncb20", "ncb20b", "noncvxu2", "noncvxun",
             "nondia", "nondquar", "nzf1", "penalty2", "penalty3", "powellsg", "power",
             "quartc", "sbrybnd", "schmvett", "scosine", "sparsine", "sparsqur", "srosenbr",
             "sinquad", "tointgss", "tquartic", "tridia", "vardim", "woods"]

#List of problems used in tests
list_problems = union(problems, problems2[1:4])

for pb in list_problems
    include("problems/$(lowercase(pb)).jl")
end

for pb in list_problems
    
  @testset "Test for $(pb)" begin
    pb_radnlp = eval(Meta.parse("$(pb)_radnlp()"))
    pb_adnlp = eval(Meta.parse("$(pb)_autodiff()"))
    
    check_nlp_dimensions(pb_radnlp)
    #multiple_precision(pb_radnlp)
    
    nlps = [pb_radnlp, pb_adnlp]
    consistent_nlps(nlps) #include derivative_check
    
    @test pb_radnlp.meta.nvar == pb_adnlp.meta.nvar

    x = rand(pb_radnlp.meta.nvar)
    @test obj(pb_radnlp, x) ≈ obj(pb_adnlp, x)
    #@test grad(pb_radnlp, x) ≈ grad(pb_adnlp, x)
    #@test hess(pb_radnlp, x) ≈ hess(pb_adnlp, x)

    v = rand(pb_radnlp.meta.nvar)
    #@test hprod(pb_radnlp, x, v) ≈ hprod(pb_adnlp, x, v)
  end
end

#include("test_memory_of_coord.jl") #TODO
#test_memory_of_coord()

#II. Tests different sparsity patterns
