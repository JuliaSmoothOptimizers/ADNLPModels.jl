# script that tests ADNLPModels over OptimizationProblems.jl problems

# optional deps
# using Enzyme
# using SparseDiffTools
using Symbolics

# AD deps
using ForwardDiff, ReverseDiff

# JSO packages
using ADNLPModels, OptimizationProblems, NLPModels, Test

# Comparison with JuMP
using JuMP, NLPModelsJuMP

names = OptimizationProblems.meta[!, :name]

for pb in names
  @info pb

  nlp = try
    OptimizationProblems.ADNLPProblems.eval(Meta.parse(pb))(backend = :default, show_time = true)
  catch e
    println("Error $e with ADNLPModel")
    continue
  end

  jum = try
    MathOptNLPModel(OptimizationProblems.PureJuMP.eval(Meta.parse(pb))())
  catch e
    println("Error $e with JuMP")
    continue
  end

  n, m = nlp.meta.nvar, nlp.meta.ncon
  x = 10 * [-(-1.0)^i for i = 1:n] # find a better point in the domain.
  v = 10 * [-(-1.0)^i for i = 1:n]
  y = 3.14 * ones(m)

  # test the main functions in the API
  try
    @testset "Test NLPModel API $(nlp.meta.name)" begin
      @test grad(nlp, x) ≈ grad(jum, x)
      @test hess(nlp, x) ≈ hess(jum, x)
      @test hess(nlp, x, y) ≈ hess(jum, x, y)
      @test hprod(nlp, x, v) ≈ hprod(jum, x, v)
      @test hprod(nlp, x, y, v) ≈ hprod(jum, x, y, v)
      if nlp.meta.ncon > 0
        @test jac(nlp, x) ≈ jac(jum, x)
        @test jprod(nlp, x, v) ≈ jprod(jum, x, v)
        @test jtprod(nlp, x, y) ≈ jtprod(jum, x, y)
      end
    end
  catch e
    println("Error $e with API")
    continue
  end
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
