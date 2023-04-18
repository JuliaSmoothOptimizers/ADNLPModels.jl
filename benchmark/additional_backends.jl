using Enzyme
# Enzyme gradient
struct EnzymeADGradient <: ADNLPModels.ADBackend end
function EnzymeADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  return EnzymeADGradient()
end
function ADNLPModels.gradient!(::EnzymeADGradient, g, f, x)
  autodiff(Reverse, f, Duplicated(x, g)) # gradient!(Reverse, g, f, x)
  return g
end

#=
using ADNLPModels, OptimizationProblems.ADNLPProblems, NLPModels
for pb in scalable_problems
  @info pb
  (pb in ["elec", "brybnd", "clplatea", "clplateb", "clplatec", "curly", "curly10", "curly20", "curly30", "ncb20", "ncb20b", "sbrybnd"]) && continue
  nlp = eval(Meta.parse(pb))(gradient_backend = EnzymeADGradient)
  grad(nlp, get_x0(nlp))
end
=#

#=
using Nabla # No longer maintained
struct NablaADGradient{T} <: ADNLPModels.ADBackend
  ∇f::T
end
function NablaADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  ∇f = Nabla.∇(f)
  return NablaADGradient(∇f)
end
function ADNLPModels.gradient!(b::NablaADGradient, g, f, x)
  g = b.∇f(x) # https://github.com/invenia/Nabla.jl/issues/61
  return g
end # doesn't work on any problem...
=#

# Generic ReverseDiff gradient
struct GenericReverseDiffADGradient <: ADNLPModels.ADBackend end
function GenericReverseDiffADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  return GenericReverseDiffADGradient()
end
function ADNLPModels.gradient!(::GenericReverseDiffADGradient, g, f, x)
  return ReverseDiff.gradient!(g, f, x)
end

struct GenericForwardDiffADGradient <: ADNLPModels.ADBackend end
function GenericForwardDiffADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  return GenericForwardDiffADGradient()
end
ADNLPModels.gradient(::GenericForwardDiffADGradient, f, x) = ForwardDiff.gradient(f, x)
function ADNLPModels.gradient!(::GenericForwardDiffADGradient, g, f, x)
  return ForwardDiff.gradient!(g, f, x)
end

using SparseDiffTools
# Hprod using ForwardDiff and SparseDiffTools
# branch hprod2: https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/pull/122/files
struct OptForwardDiffADHvprod{T, F} <: ADNLPModels.ADBackend
  tmp_in::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  tmp_out::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  ϕ!::F
end
function OptForwardDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  tmp_in = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  tmp_out = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  cfg = ForwardDiff.GradientConfig(f, tmp_in)
  ϕ!(dy, x; f = f, cfg = cfg) = ForwardDiff.gradient!(dy, f, x, cfg)
  return OptForwardDiffADHvprod(tmp_in, tmp_out, ϕ!)
end

function ADNLPModels.Hvprod!(b::OptForwardDiffADHvprod, Hv, f, x, v)
  ϕ!(dy, x; f = f) = ForwardDiff.gradient!(dy, f, x)
  SparseDiffTools.auto_hesvecgrad!(Hv, ϕ!, x, v, b.tmp_in, b.tmp_out)
  return Hv
end

#
#
struct OptimizedReverseDiffADJtprod{T, S, GT} <: ADNLPModels.InPlaceADbackend
  #ψ::F2
  #gcfg::GC  # gradient config
  gtape::GT  # compiled gradient tape
  _tmp_out::Vector{ReverseDiff.TrackedReal{T, T, Nothing}}
  _rval::S  # temporary storage for jtprod
end

function OptimizedReverseDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  y0::AbstractVector{T} = rand(ncon),
  kwargs...,
) where {T}
  _tmp_out = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, ncon)
  _rval = similar(x0, ncon)  # temporary storage for jtprod

  ψ(x, u; tmp_out = _tmp_out) = begin
    # here x is a vector of ReverseDiff.TrackedReal
    c!(tmp_out, x)
    dot(tmp_out, u)
  end

  # u = similar(x0, nequ)  # just for GradientConfig
  gcfg = ReverseDiff.GradientConfig((x0, y0))
  gtape = ReverseDiff.compile(ReverseDiff.GradientTape(ψ, (x0, y0), gcfg))

  return OptimizedReverseDiffADJtprod(gtape, _tmp_out, _rval)
end

function ADNLPModels.Jtprod!(b::OptimizedReverseDiffADJtprod, Jtv, c!, x, v)
  ReverseDiff.gradient!((Jtv, b._rval), b.gtape, (x, v))
  return Jtv
end

using ForwardDiff, SparseDiffTools
struct OptimizedForwardDiffADJtprod{Tag, GT, S} <: ADNLPModels.InPlaceADbackend
  cfg::ForwardDiff.GradientConfig{Tag}
  ψ::GT
  _tmp_out::S
end

function OptimizedForwardDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  y0::AbstractVector{T} = rand(eltype(x0), ncon),
  kwargs...,
) where {T}
  function ψ(x; u = y0, tmp_out = similar(x, ncon))
    c!(tmp_out, x)
    dot(tmp_out, u)
  end

  tag = ForwardDiff.Tag(ψ, typeof(x0))
  cfg = ForwardDiff.GradientConfig(ψ, x0, ForwardDiff.Chunk(x0), tag)

  _tmp_out = similar(cfg.duals, ncon)

  return OptimizedForwardDiffADJtprod(cfg, ψ, _tmp_out)
end

function ADNLPModels.Jtprod!(b::OptimizedForwardDiffADJtprod{Tag, T, GT}, Jtv, c!, x, v) where {Tag, T, GT}
  ydual = ForwardDiff.vector_mode_dual_eval!(x -> b.ψ(x, u = v, tmp_out = b._tmp_out), b.cfg, x)
  #=
    ydual, xdual = cfg.duals
    seed!(xdual, x, cfg.seeds)
    seed!(ydual, y)
    f!(ydual, xdual)
  =#
  ForwardDiff.extract_gradient!(Tag, Jtv, ydual)
  return Jtv
end

#=
using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems, NLPModels

const meta = OptimizationProblems.meta
const nn = OptimizationProblems.default_nvar # 100 # default parameter for scalable problems

# Scalable problems from OptimizationProblem.jl
scalable_problems = meta[meta.variable_nvar .== true, :name] # problems that are scalable

all_problems = meta[meta.nvar .> 5, :name] # all problems with ≥ 5 variables
all_problems = setdiff(all_problems, scalable_problems) # avoid duplicate problems

all_cons_problems = meta[(meta.nvar .> 5) .&& (meta.ncon .> 5), :name] # all problems with ≥ 5 variables
scalable_cons_problems = meta[(meta.variable_nvar .== true) .&& (meta.ncon .> 5), :name] # problems that are scalable
all_cons_problems = setdiff(all_cons_problems, scalable_cons_problems) # avoid duplicate problems

for pb in scalable_cons_problems[1:1]
  @info pb
  (pb in []) && continue
  nlp = eval(Meta.parse(pb))(jtprod_backend = OptimizedForwardDiffADJtprod)
  jtprod(nlp, get_x0(nlp), get_y0(nlp))
end
=#
