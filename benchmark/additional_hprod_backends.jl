#=
using ADNLPModels, OptimizationProblems.ADNLPProblems, NLPModels
for pb in scalable_problems
  n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
  @info " $(pb): $n vars and $m cons"
  v = [sin(T(i) / 10) for i=1:n]
  (pb in [""]) && continue
  nlp = eval(Meta.parse(pb))(hprod_backend = EnzymeADGradient)
  x = get_x0(nlp)
  Hv_control = ForwardDiff.hessian(nlp.f, x) * v
  hprod!(nlp, x, v, Hv)
end
=#

using SparseDiffTools
# Hprod using ForwardDiff and SparseDiffTools
# branch hprod2: https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/pull/122/files
struct SPTADHvprod{T, F} <: ADNLPModels.ADBackend
  tmp_in::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  tmp_out::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  ϕ!::F
end
function SPTADHvprod(
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
  return SPTADHvprod(tmp_in, tmp_out, ϕ!)
end

function ADNLPModels.Hvprod!(b::SPTADHvprod, Hv, f, x, v)
  # ϕ!(dy, x; f = f) = ForwardDiff.gradient!(dy, f, x)
  SparseDiffTools.auto_hesvecgrad!(Hv, b.ϕ!, x, v, b.tmp_in, b.tmp_out)
  return Hv
end

#=
using ADNLPModels, OptimizationProblems.ADNLPProblems, NLPModels, Test
T = Float64
nscal = 32
@testset "$pb" for pb in scalable_problems
  n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
  v = [sin(T(i) / 10) for i=1:n]
  (pb in [""]) && continue
  nlp = eval(Meta.parse(pb))(n = nscal, hprod_backend = SPTADHvprod)
  @info " $(pb): $n vars ($(nlp.meta.nvar)) and $m cons"
  x = get_x0(nlp)
  Hv_control = ForwardDiff.hessian(nlp.f, x) * v
  Hv = similar(x)
  hprod!(nlp, x, v, Hv)
  @test Hv ≈ Hv_control
end
=#

#######################################################
# ForwardDiff

struct ForwardDiffADHvprod1{T, F} <: ADNLPModels.ADBackend
  z::Vector{ForwardDiff.Dual{Nothing, T, 1}}
  gz::Vector{ForwardDiff.Dual{Nothing, T, 1}}
  ∇φ::F
end

function ForwardDiffADHvprod1(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  tag = Nothing

  z = Vector{ForwardDiff.Dual{tag, T, 1}}(undef, nvar)
  gz = Vector{ForwardDiff.Dual{tag, T, 1}}(undef, nvar)
  cfg = ForwardDiff.GradientConfig(f, z)
  ∇φ(gz, z; f = f, cfg = cfg) = ForwardDiff.gradient!(gz, f, z, cfg) # ∇f(x + ε * v) = ∇f(x) + ε * ∇²f(x)ᵀv

  return ForwardDiffADHvprod1(z, gz, ∇φ)
end

function ADNLPModels.Hvprod!(b::ForwardDiffADHvprod1, Hv, f, x, v)
  map!(ForwardDiff.Dual, b.z, x, v) # x + ε * v
  b.∇φ(b.gz, b.z)
  ForwardDiff.extract_derivative!(Nothing, Hv, b.gz)  # ∇²f(x)ᵀv
  return Hv
end

#=
using ADNLPModels, OptimizationProblems.ADNLPProblems, NLPModels, Test
T = Float64
nscal = 32
@testset "$pb" for pb in scalable_problems
  n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
  v = [sin(T(i) / 10) for i=1:n]
  (pb in [""]) && continue
  nlp = eval(Meta.parse(pb))(n = nscal, hprod_backend = ForwardDiffADHvprod1)
  @info " $(pb): $n vars ($(nlp.meta.nvar)) and $m cons"
  x = get_x0(nlp)
  Hv_control = ForwardDiff.hessian(nlp.f, x) * v
  Hv = similar(x)
  hprod!(nlp, x, v, Hv)
  @test Hv ≈ Hv_control
end
=#

struct ForwardDiffADHvprod2{T, F} <: ADNLPModels.ADBackend
  z::Vector{ForwardDiff.Dual{Nothing, T, 1}}
  gz::Vector{ForwardDiff.Dual{Nothing, T, 1}}
  ∇φ::F
end
function ForwardDiffADHvprod2(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  tag = Nothing

  z = Vector{ForwardDiff.Dual{tag, T, 1}}(undef, nvar)
  gz = Vector{ForwardDiff.Dual{tag, T, 1}}(undef, nvar)
  cfg = ForwardDiff.GradientConfig(f, z)
  ∇φ(gz, z; f = f, cfg = cfg) = ForwardDiff.gradient!(gz, f, z, cfg) # ∇f(x + ε * v) = ∇f(x) + ε * ∇²f(x)ᵀv

  return ForwardDiffADHvprod2(z, gz, ∇φ)
end

function ADNLPModels.Hvprod!(b::ForwardDiffADHvprod2, Hv, f, x, v)
  map!(ForwardDiff.Dual, b.z, x, v) # x + ε * v
  b.∇φ(b.gz, b.z)
  ForwardDiff.extract_derivative!(Nothing, Hv, b.gz)  # ∇²f(x)ᵀv
  return Hv
end

#=
using ADNLPModels, OptimizationProblems.ADNLPProblems, NLPModels, Test
T = Float64
nscal = 32
@testset "$pb" for pb in scalable_problems
  n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
  v = [sin(T(i) / 10) for i=1:n]
  (pb in [""]) && continue
  nlp = eval(Meta.parse(pb))(n = nscal, hprod_backend = ForwardDiffADHvprod2)
  @info " $(pb): $n vars ($(nlp.meta.nvar)) and $m cons"
  x = get_x0(nlp)
  Hv_control = ForwardDiff.hessian(nlp.f, x) * v
  Hv = similar(x)
  hprod!(nlp, x, v, Hv)
  @test Hv ≈ Hv_control
end
=#

struct ForwardDiffADHvprod3{T, F} <: ADNLPModels.ADBackend
  z::Vector{ForwardDiff.Dual{Nothing, T, 1}}
  gz::Vector{ForwardDiff.Dual{Nothing, T, 1}}
  cfg::F
end
function ForwardDiffADHvprod3(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  z = Vector{ForwardDiff.Dual{Nothing, T, 1}}(undef, nvar)
  gz = Vector{ForwardDiff.Dual{Nothing, T, 1}}(undef, nvar)
  cfg = ForwardDiff.GradientConfig(f, z)
  return ForwardDiffADHvprod3(z, gz, cfg)
end

function ADNLPModels.Hvprod!(b::ForwardDiffADHvprod3, Hv, f, x, v)
  map!(ForwardDiff.Dual, b.z, x, v) # x + ε * v
  ForwardDiff.vector_mode_gradient!(b.gz, f, b.z, b.cfg)
  ForwardDiff.extract_derivative!(Nothing, Hv, b.gz)  # ∇²f(x)ᵀv
  return Hv
end

#=
using ADNLPModels, OptimizationProblems.ADNLPProblems, NLPModels, Test
T = Float64
@testset "$pb" for pb in scalable_problems
  n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
  @info " $(pb): $n vars and $m cons"
  v = [sin(T(i) / 10) for i=1:n]
  (pb in [""]) && continue
  nlp = eval(Meta.parse(pb))(hprod_backend = ForwardDiffADHvprod3)
  x = get_x0(nlp)
  Hv_control = ForwardDiff.hessian(nlp.f, x) * v
  Hv = similar(x)
  hprod!(nlp, x, v, Hv)
  @test Hv ≈ Hv_control
end
# doesn't work...
=#

struct ForwardDiffADHvprod4{T, F} <: ADNLPModels.ADBackend
  z::Vector{ForwardDiff.Dual{Nothing, T, 1}}
  gz::Vector{ForwardDiff.Dual{Nothing, T, 1}}
  ∇φ::F
end
function ForwardDiffADHvprod4(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  z = Vector{ForwardDiff.Dual{Nothing, T, 1}}(undef, nvar)
  gz = Vector{ForwardDiff.Dual{Nothing, T, 1}}(undef, nvar)
  ∇φ(gz, z) = ForwardDiff.gradient!(gz, f, z)
  return ForwardDiffADHvprod4(z, gz, ∇φ)
end

function ADNLPModels.Hvprod!(b::ForwardDiffADHvprod4, Hv, f, x, v)
  map!(ForwardDiff.Dual, b.z, x, v) # x + ε * v
  b.∇φ(b.gz, b.z) # ∇f(x + ε * v) = ∇f(x) + ε * ∇²f(x)ᵀv
  ForwardDiff.extract_derivative!(Nothing, Hv, b.gz)  # ∇²f(x)ᵀv
  return Hv
end

#=
using ADNLPModels, OptimizationProblems.ADNLPProblems, NLPModels, Test
T = Float64
nscal = 32
@testset "$pb" for pb in scalable_problems
  n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
  v = [sin(T(i) / 10) for i=1:n]
  (pb in [""]) && continue
  nlp = eval(Meta.parse(pb))(n = nscal, hprod_backend = ForwardDiffADHvprod4)
  @info " $(pb): $n vars ($(nlp.meta.nvar)) and $m cons"
  x = get_x0(nlp)
  Hv_control = ForwardDiff.hessian(nlp.f, x) * v
  Hv = similar(x)
  hprod!(nlp, x, v, Hv)
  @test Hv ≈ Hv_control
end
=#

struct ForwardDiffADHvprod5{T, Tag, F} <: ADNLPModels.ADBackend
  z::Vector{ForwardDiff.Dual{Tag, T, 1}}
  gz::Vector{ForwardDiff.Dual{Tag, T, 1}}
  ∇φ::F
end
function ForwardDiffADHvprod5(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  tag = ForwardDiff.Tag{typeof(f), T}
  z = Vector{ForwardDiff.Dual{tag, T, 1}}(undef, nvar)
  gz = similar(z)
  cfg = ForwardDiff.GradientConfig(f, z)
  ∇φ!(gz, z; f = f, cfg = cfg) = ForwardDiff.gradient!(gz, f, z, cfg)
  return ForwardDiffADHvprod5(z, gz, ∇φ!)
end

function ADNLPModels.Hvprod!(b::ForwardDiffADHvprod5{T, Tag, F}, Hv, f, x, v) where {T, Tag, F}
  map!(ForwardDiff.Dual{Tag}, b.z, x, v) # x + ε * v
  b.∇φ(b.gz, b.z) # ∇f(x + ε * v) = ∇f(x) + ε * ∇²f(x)ᵀv
  ForwardDiff.extract_derivative!(Tag, Hv, b.gz)  # ∇²f(x)ᵀv
  return Hv
end

#=
using ADNLPModels, OptimizationProblems.ADNLPProblems, NLPModels, Test
T = Float64
nscal = 32
@testset "$pb" for pb in scalable_problems
  n = eval(Meta.parse("OptimizationProblems.get_" * pb * "_nvar(n = $(nscal))"))
  m = eval(Meta.parse("OptimizationProblems.get_" * pb * "_ncon(n = $(nscal))"))
  v = [sin(T(i) / 10) for i=1:n]
  (pb in [""]) && continue
  nlp = eval(Meta.parse(pb))(n = nscal, hprod_backend = ForwardDiffADHvprod5)
  @info " $(pb): $n vars ($(nlp.meta.nvar)) and $m cons"
  x = get_x0(nlp)
  Hv_control = ForwardDiff.hessian(nlp.f, x) * v
  Hv = similar(x)
  hprod!(nlp, x, v, Hv)
  @test Hv ≈ Hv_control
end
=#

#######################################################
# ReverseDiff
#
#
