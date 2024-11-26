using LinearAlgebra, SparseArrays, Test
using ADNLPModels, ManualNLPModels, NLPModels, NLPModelsModifiers, NLPModelsTest
using ADNLPModels:
  gradient, gradient!, jacobian, hessian, Jprod!, Jtprod!, directional_second_derivative, Hvprod!

for problem in NLPModelsTest.nlp_problems ∪ ["GENROSE"]
  include("nlp/problems/$(lowercase(problem)).jl")
end
for problem in NLPModelsTest.nls_problems
  include("nls/problems/$(lowercase(problem)).jl")
end

#=
ADNLPModels.EmptyADbackend(args...; kwargs...) = ADNLPModels.EmptyADbackend()

names = OptimizationProblems.meta[!, :name]
list_excluded_enzyme = [
  "brybnd",
  "clplatea",
  "clplateb",
  "clplatec",
  "curly",
  "curly10",
  "curly20",
  "curly30",
  "elec",
  "fminsrf2",
  "hs101",
  "hs117",
  "hs119",
  "hs86",
  "integreq",
  "ncb20",
  "ncb20b",
  "palmer1c",
  "palmer1d",
  "palmer2c",
  "palmer3c",
  "palmer4c",
  "palmer5c",
  "palmer5d",
  "palmer6c",
  "palmer7c",
  "palmer8c",
  "sbrybnd",
  "tetra",
  "tetra_duct12",
  "tetra_duct15",
  "tetra_duct20",
  "tetra_foam5",
  "tetra_gear",
  "tetra_hook",
  "threepk",
  "triangle",
  "triangle_deer",
  "triangle_pacman",
  "triangle_turtle",
  "watson",
]
for pb in names
  @info pb
  (pb in list_excluded_enzyme) && continue
  nlp = eval(Meta.parse(pb))(
    gradient_backend = ADNLPModels.EnzymeADGradient,
    jacobian_backend = ADNLPModels.EmptyADbackend,
    hessian_backend = ADNLPModels.EmptyADbackend,
  )
  grad(nlp, get_x0(nlp))
end
=#

#=
ERROR: Duplicated Returns not yet handled
Stacktrace:
 [1] autodiff
   @.julia\packages\Enzyme\DIkTv\src\Enzyme.jl:209 [inlined]
 [2] autodiff(mode::EnzymeCore.ReverseMode, f::OptimizationProblems.ADNLPProblems.var"#f#254"{OptimizationProblems.ADNLPProblems.var"#f#250#255"}, args::Duplicated{Vector{Float64}})
   @ Enzyme.julia\packages\Enzyme\DIkTv\src\Enzyme.jl:248
 [3] gradient!(#unused#::ADNLPModels.EnzymeADGradient, g::Vector{Float64}, f::Function, x::Vector{Float64})
   @ ADNLPModelsDocuments\cvs\ADNLPModels.jl\src\enzyme.jl:17
 [4] grad!(nlp::ADNLPModel{Float64, Vector{Float64}, Vector{Int64}}, x::Vector{Float64}, g::Vector{Float64})
   @ ADNLPModelsDocuments\cvs\ADNLPModels.jl\src\nlp.jl:542
 [5] grad(nlp::ADNLPModel{Float64, Vector{Float64}, Vector{Int64}}, x::Vector{Float64})
   @ NLPModels.julia\packages\NLPModels\XBcWL\src\nlp\api.jl:31
 [6] top-level scope
   @ .\REPL[7]:5
=#
