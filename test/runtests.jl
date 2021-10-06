using ADNLPModels, ForwardDiff, LinearAlgebra, NLPModels, NLPModelsModifiers, NLPModelsTest, Test

for problem in NLPModelsTest.nlp_problems ∪ ["GENROSE"]
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in NLPModelsTest.nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end

# Automatically loads the code for Zygote and ReverseDiff with Requires
import Zygote, ReverseDiff
using ADNLPModels: ForwardDiffAD, ZygoteAD, ReverseDiffAD

include("nlp/basic.jl")
include("nls/basic.jl")
include("nlp/nlpmodelstest.jl")
include("nls/nlpmodelstest.jl")
