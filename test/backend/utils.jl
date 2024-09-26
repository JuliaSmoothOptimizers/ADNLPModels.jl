using OptimizationProblems
using NLPModels

ADNLPModels.EmptyADbackend(args...; kwargs...) = ADNLPModels.EmptyADbackend()

function test_adbackend(package::Symbol)
  names = OptimizationProblems.meta[!, :name]
  for pb in names
    @info pb
    nlp = OptimizationProblems.ADNLPProblems.eval(Meta.parse(pb))(backend=package)
    grad(nlp, nlp.meta.x0)
  end
end
