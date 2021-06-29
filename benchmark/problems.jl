using Pkg
Pkg.add(url="https://github.com/tmigot/ADNLPModelProblems")
using ADNLPModelProblems, NLPModelsJuMP

# Scalable problems from ADNLPModelProblems.jl
const problems = ["clnlbeam", "controlinvestment", "hovercraft1d", "polygon1", "polygon2", "polygon3"]

nn = 100 # default parameter for scalable problems
for pb in problems
  nlp = eval(
    Meta.parse(
      "ADNLPModelProblems.$(pb)_autodiff(n=$(nn))",
    ),
  )
  nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
  eval(
    Meta.parse(
      "$(pb)_reverse(args... ; kwargs...) = ADNLPModelProblems.$(pb)_autodiff(args... ; adbackend=ADNLPModels.ReverseDiffAD($(nvar), $(ncon)), n=$(nn), kwargs...)",
    ),
  )
  eval(
    Meta.parse(
      "$(pb)_zygote(args... ; kwargs...) = ADNLPModelProblems.$(pb)_autodiff(args... ; adbackend=ADNLPModels.ZygoteAD($(nvar), $(ncon)), n=$(nn), kwargs...)",
    ),
  )
  eval(
    Meta.parse(
      "$(pb)_jump(args... ; n=$(nn), kwargs...) = MathOptNLPModel(ADNLPModelProblems.$(pb)(n))",
    ),
  )
end