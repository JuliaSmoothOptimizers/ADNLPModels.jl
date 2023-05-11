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
using ReverseDiff
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
