#
# Je ne sais pas si c'est un bug, mais c'est bizarre comme situation:
#
#
#
using NLPModels, Test, Zygote

problems = ["penalty3"]

#function ∇f!(g, x)
#    g .= Zygote.gradient(nlp.f, x)[1]
#    return g
#end

nlp = penalty3_radnlp()
g = Zygote.gradient(nlp.f, nlp.meta.x0)

@test typeof(g) <: Tuple
@test size(g[1]) == size(nlp.meta.x0)