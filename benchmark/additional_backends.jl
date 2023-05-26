include("additional_hprod_backends.jl")

using ADNLPModels, OptimizationProblems.ADNLPProblems, NLPModels

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

##########################################################

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

#=
using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems, NLPModels, Test

const meta = OptimizationProblems.meta
const nn = OptimizationProblems.default_nvar # 100 # default parameter for scalable problems

# Scalable problems from OptimizationProblem.jl
scalable_problems = meta[meta.variable_nvar .== true, :name] # problems that are scalable

all_problems = meta[meta.nvar .> 5, :name] # all problems with ≥ 5 variables
all_problems = setdiff(all_problems, scalable_problems) # avoid duplicate problems

all_cons_problems = meta[(meta.nvar .> 5) .&& (meta.ncon .> 5), :name] # all problems with ≥ 5 variables
scalable_cons_problems = meta[(meta.variable_nvar .== true) .&& (meta.ncon .> 5), :name] # problems that are scalable
all_cons_problems = setdiff(all_cons_problems, scalable_cons_problems) # avoid duplicate problems

T = Float64
nscal = 32
for pb in scalable_cons_problems
  # (pb in ["hovercraft1d", "polygon1", "structural"]) && continue
  n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
  nlp = eval(Meta.parse(pb))(n = nscal, jtprod_backend = OptimizedReverseDiffADJtprod)
  @info " $(pb): $n vars ($(nlp.meta.nvar)) and $m cons"
  ncon = nlp.meta.nnln
  x = get_x0(nlp)
  cx = similar(x, ncon)
  v = T[sin(T(i) / 10) for i=1:ncon]
  Jtv_control = ForwardDiff.jacobian(nlp.c!, cx, x)' * v
  Jtv = similar(x, n)
  jtprod_nln!(nlp, x, v, Jtv)
  @test Jtv ≈ Jtv_control
end
=#

struct OptimizedReverseDiffADJtprod2{T, S, GT} <: ADNLPModels.InPlaceADbackend
  #ψ::F2
  #gcfg::GC  # gradient config
  gtape::GT  # compiled gradient tape
  _rval::S  # temporary storage for jtprod
  _cx::S
end

function OptimizedReverseDiffADJtprod2( # ERROR: TrackedArrays do not support setindex!
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  y0::AbstractVector{T} = rand(ncon),
  kwargs...,
) where {T}
  _rval = similar(x0, ncon)  # temporary storage for jtprod
  _cx = similar(y0)

  function ψ(cx, x, u)
    c!(cx, x)
    dot(cx, u)
  end

  gcfg = ReverseDiff.GradientConfig((y0, x0, y0))
  gtape = ReverseDiff.compile(ReverseDiff.GradientTape(ψ, (y0, x0, y0), gcfg))

  return OptimizedReverseDiffADJtprod2(gtape, _rval, _cx)
end

function ADNLPModels.Jtprod!(b::OptimizedReverseDiffADJtprod2, Jtv, c!, x, v)
  ReverseDiff.gradient!((b._cx, Jtv, b._rval), b.gtape, (b._cx, x, v))
  return Jtv
end

#=
using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems, NLPModels, Test

const meta = OptimizationProblems.meta
const nn = OptimizationProblems.default_nvar # 100 # default parameter for scalable problems

# Scalable problems from OptimizationProblem.jl
scalable_problems = meta[meta.variable_nvar .== true, :name] # problems that are scalable

all_problems = meta[meta.nvar .> 5, :name] # all problems with ≥ 5 variables
all_problems = setdiff(all_problems, scalable_problems) # avoid duplicate problems

all_cons_problems = meta[(meta.nvar .> 5) .&& (meta.ncon .> 5), :name] # all problems with ≥ 5 variables
scalable_cons_problems = meta[(meta.variable_nvar .== true) .&& (meta.ncon .> 5), :name] # problems that are scalable
all_cons_problems = setdiff(all_cons_problems, scalable_cons_problems) # avoid duplicate problems

T = Float64
nscal = 32
for pb in scalable_cons_problems
  # (pb in ["hovercraft1d", "polygon1", "structural"]) && continue
  n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
  nlp = eval(Meta.parse(pb))(n = nscal, jtprod_backend = OptimizedReverseDiffADJtprod2)
  @info " $(pb): $n vars ($(nlp.meta.nvar)) and $m cons"
  ncon = nlp.meta.nnln
  x = get_x0(nlp)
  cx = similar(x, ncon)
  v = T[sin(T(i) / 10) for i=1:ncon]
  Jtv_control = ForwardDiff.jacobian(nlp.c!, cx, x)' * v
  Jtv = similar(x, n)
  jtprod_nln!(nlp, x, v, Jtv)
  @test Jtv ≈ Jtv_control
end
=#

using ForwardDiff
struct OptimizedForwardDiffADJtprod{Tag, GT, S} <: ADNLPModels.InPlaceADbackend
  cfg::ForwardDiff.GradientConfig{Tag}
  ψ::GT
  temp::S
  sol::S
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

  temp = similar(x0, nvar + 2 * ncon)
  sol = similar(x0, nvar + 2 * ncon)

  function ψ(z; nvar = nvar, ncon = ncon)
    cx, x, u = view(z, 1:ncon), view(z, (ncon + 1):(nvar + ncon)), view(z, (nvar + ncon + 1):(nvar + ncon + ncon))
    c!(cx, x)
    dot(cx, u)
  end
  tagψ = ForwardDiff.Tag(ψ, T)
  cfg = ForwardDiff.GradientConfig(ψ, temp, ForwardDiff.Chunk(temp), tagψ)

  return OptimizedForwardDiffADJtprod(cfg, ψ, temp, sol)
end

function ADNLPModels.Jtprod!(b::OptimizedForwardDiffADJtprod{Tag, GT, S}, Jtv, c!, x, v) where {Tag, GT, S}
  ncon = length(v)
  nvar = length(x)

  b.sol[(ncon + 1):(ncon + nvar)] .= x
  b.sol[(ncon + nvar + 1):(2 * ncon + nvar)] .= v
  ForwardDiff.gradient!(b.temp, b.ψ, b.sol, b.cfg)
  Jtv .= view(b.temp, (ncon + 1):(nvar + ncon))
  return Jtv
end

#= First Attempt: not successful
nvar = 5
ncon = 1
T = Float64
x = ones(T, nvar)
Jtv = similar(x)
cx = similar(x, ncon)
v = rand(T, ncon)
function c!(cx, x)
  cx[1] = x[2]
  cx
end
tagc = ForwardDiff.Tag(c!, T)
function ψ(z; nvar = nvar, ncon = ncon)
  cx, u = view(z, 1:ncon), view(z, (ncon + 1):(ncon + ncon))
  dot(cx, u)
end
tagψ = ForwardDiff.Tag(ψ, T)
Tag = typeof(tagψ)
cfg = ForwardDiff.GradientConfig(ψ, vcat(cx, v), ForwardDiff.Chunk(vcat(cx, v)), tagψ)

z = Vector{ForwardDiff.Dual{Tag, T, 1}}(x)
map!(ForwardDiff.Dual{Tag}, z, x, ones(T, nvar))
cz = similar(z, ncon)
c!(cz, z)
u = Vector{ForwardDiff.Dual{Tag, T, 1}}(v)
map!(ForwardDiff.Dual{Tag}, u, v, zeros(T, ncon))
tmp = ψ(vcat(cz, u))

Jtv = zeros(1)
Jtv .= 0
ForwardDiff.extract_gradient!(Tag, Jtv, ψ(vcat(cz, u)))
=#

#=
using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems, NLPModels, Test

const meta = OptimizationProblems.meta
const nn = OptimizationProblems.default_nvar # 100 # default parameter for scalable problems

# Scalable problems from OptimizationProblem.jl
scalable_problems = meta[meta.variable_nvar .== true, :name] # problems that are scalable

all_problems = meta[meta.nvar .> 5, :name] # all problems with ≥ 5 variables
all_problems = setdiff(all_problems, scalable_problems) # avoid duplicate problems

all_cons_problems = meta[(meta.nvar .> 5) .&& (meta.ncon .> 5), :name] # all problems with ≥ 5 variables
scalable_cons_problems = meta[(meta.variable_nvar .== true) .&& (meta.ncon .> 5), :name] # problems that are scalable
all_cons_problems = setdiff(all_cons_problems, scalable_cons_problems) # avoid duplicate problems

T = Float64
nscal = 32
for pb in scalable_cons_problems
  # (pb in ["hovercraft1d", "polygon1", "structural"]) && continue
  n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
  nlp = eval(Meta.parse(pb))(n = nscal, jtprod_backend = OptimizedForwardDiffADJtprod)
  @info " $(pb): $n vars ($(nlp.meta.nvar)) and $m cons"
  ncon = nlp.meta.nnln
  x = get_x0(nlp)
  cx = similar(x, ncon)
  v = T[sin(T(i) / 10) for i=1:ncon]
  Jtv_control = ForwardDiff.jacobian(nlp.c!, cx, x)' * v
  Jtv = similar(x, n)
  jtprod_nln!(nlp, x, v, Jtv)
  @test Jtv ≈ Jtv_control
end
=#
