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
