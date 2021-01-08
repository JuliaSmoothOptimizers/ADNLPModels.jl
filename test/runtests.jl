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

#List of problems used in tests
#Problems from NLPModels
include("problems/hs5.jl") #bounds constraints n=2, dense hessian
include("problems/brownden.jl") #unconstrained n=4, dense hessian

for pb in union(problems, problems2)
    include("problems/$(lowercase(pb)).jl")
end

problems = ["hs5", "brownden"]
problems2 = ["arglina", "arwhead", "chainwoo"]

for pb in union(problems, problems2)
    
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