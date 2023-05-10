include("additional_hprod_backends.jl")

using ADNLPModels, OptimizationProblems.ADNLPProblems, NLPModels
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

#= Doesn't work with Enzyme 0.11 ...
for pb in scalable_problems
  @info pb
  # (pb in ["elec", "brybnd", "clplatea", "clplateb", "clplatec", "curly", "curly10", "curly20", "curly30", "ncb20", "ncb20b", "sbrybnd"]) && continue
  (pb in ["NZF1"]) && continue
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

##########################################################
using ForwardDiff

struct ForwardDiffADJprod1{T, Tag} <: ADNLPModels.InPlaceADbackend
  z::Vector{ForwardDiff.Dual{Tag, T, 1}}
  cz::Vector{ForwardDiff.Dual{Tag, T, 1}}
end

function ForwardDiffADJprod1(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  tag = ForwardDiff.Tag{typeof(c!), T}
  
  z = Vector{ForwardDiff.Dual{tag, T, 1}}(undef, nvar)
  cz = similar(z, ncon)
  return ForwardDiffADJprod1(z, cz)
end

function ADNLPModels.Jprod!(b::ForwardDiffADJprod1{T, Tag}, Jv, c!, x, v) where {T, Tag}
  map!(ForwardDiff.Dual{Tag}, b.z, x, v) # x + ε * v
  c!(b.cz, b.z) # c!(cz, x + ε * v)
  ForwardDiff.extract_derivative!(Tag, Jv, b.cz) # ∇c!(cx, x)ᵀv
  return Jv
end

#=
using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems, NLPModels, Test
T = Float64
nscal = 32
@testset "$pb" for pb in scalable_cons_problems
  n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
  v = [sin(T(i) / 10) for i=1:n]
  (pb in ["chain"]) && continue
  nlp = eval(Meta.parse(pb))(n = nscal, jprod_backend = ForwardDiffADJprod1)
  @info " $(pb): $n vars ($(nlp.meta.nvar)) and $m cons"
  ncon = nlp.meta.nnln
  x = get_x0(nlp)
  cx = similar(x, ncon)
  Jv_control = ForwardDiff.jacobian(nlp.c!, cx, x) * v
  Jv = similar(x, ncon)
  jprod_nln!(nlp, x, v, Jv)
  @test Jv ≈ Jv_control
end
=#

##########################################################

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
