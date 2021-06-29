# Scalable problems from ADNLPModelProblems.jl
const problems = ["clnlbeam", "controlinvestment", "hovercraft1d", "polygon1", "polygon2", "polygon3"]

for pb in problem
  include("$(lowercase(pb)).jl")
end

nvar = 100 # default parameter for scalable problems
for pb in problems
  eval(
    Meta.parse(
      "$(pb)_reverse(args... ; kwargs...) = $(pb)_autodiff(args... ; adbackend=ADNLPModels.ReverseDiffAD(), n=$(nvar), kwargs...)",
    ),
  )
  eval(
    Meta.parse(
      "$(pb)_zygote(args... ; kwargs...) = $(pb)_autodiff(args... ; adbackend=ADNLPModels.ZygoteAD(), n=$(nvar), kwargs...)",
    ),
  )
  eval(
    Meta.parse(
      "$(pb)_jump(args... ; n=$(nvar), kwargs...) = MathOptNLPModel($(pb)(n))",
    ),
  )
  end
end