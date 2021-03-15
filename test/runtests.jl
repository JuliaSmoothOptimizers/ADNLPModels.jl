using ADNLPModels, LinearAlgebra, NLPModels, NLPModelsModifiers, NLPModelsTest, SparseArrays, Test

for problem in NLPModelsTest.nlp_problems ∪ ["GENROSE"]
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in NLPModelsTest.nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end
for problem in setdiff(NLPModelsTest.nlp_problems,["MGH01Feas", "LINCON"])
  include("problems/$(lowercase(problem)).jl") #Tangi: should be moved to radnlp/problems
end

#=
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
list_problems = problems
for pb in list_problems
    include("problems/$(lowercase(pb)).jl")
end
=#

#=
include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")
=#
#include("radnlp/basic.jl") #Doesn't work
include("radnlp/nlpmodelstest.jl") #Partially work