include("additional_hprod_backends.jl")

using Symbolics, SparseDiffTools

struct SparseADHessianSDTColoration{S} <: ADNLPModels.ADBackend
  d::BitVector
  rowval::Vector{Int}
  colptr::Vector{Int}
  colors::Vector{Int}
  ncolors::Int
  res::S
end

function SparseADHessianSDTColoration(
  nvar,
  f,
  ncon,
  c!;
  x0 = rand(nvar),
  alg::SparseDiffTools.SparseDiffToolsColoringAlgorithm = SparseDiffTools.GreedyD1Color(),
  kwargs...,
)
  Symbolics.@variables xs[1:nvar]
  xsi = Symbolics.scalarize(xs)
  fun = f(xsi)
  if ncon > 0
    Symbolics.@variables ys[1:ncon]
    ysi = Symbolics.scalarize(ys)
    cx = similar(ysi)
    fun = fun + dot(c!(cx, xsi), ysi)
  end
  S = Symbolics.hessian_sparsity(fun, ncon == 0 ? xsi : [xsi; ysi]) # , full = false
  H = ncon == 0 ? S : S[1:nvar, 1:nvar]
  colors = matrix_colors(H, alg)
  d = BitVector(undef, nvar)
  ncolors = maximum(colors)

  trilH = tril(H)
  rowval = trilH.rowval
  colptr = trilH.colptr

  res = similar(x0)

  return SparseADHessianSDTColoration(d, rowval, colptr, colors, ncolors, res)
end

function ADNLPModels.get_nln_nnzh(b::SparseADHessianSDTColoration, nvar)
  return length(b.rowval)
end

function ADNLPModels.hess_structure!(
  b::SparseADHessianSDTColoration,
  nlp::ADNLPModels.ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rowval
  for i = 1:(nlp.meta.nvar)
    for j = b.colptr[i]:(b.colptr[i + 1] - 1)
      cols[j] = i
    end
  end
  return rows, cols
end

function ADNLPModels.sparse_hess_coord!(
  ℓ::Function,
  b::SparseADHessianSDTColoration,
  x::AbstractVector,
  vals::AbstractVector,
)
  nvar = length(x)
  for icol = 1:(b.ncolors)
    b.d .= (b.colors .== icol)
    ForwardDiff.derivative!(b.res, t -> ForwardDiff.gradient(ℓ, x + t * b.d), 0)
    for j = 1:nvar
      if b.colors[j] == icol
        for k = b.colptr[j]:(b.colptr[j + 1] - 1)
          i = b.rowval[k]
          vals[k] = b.res[i]
        end
      end
    end
  end
  return vals
end

function ADNLPModels.hess_coord!(
  b::SparseADHessianSDTColoration,
  nlp::ADNLPModels.ADModel,
  x::AbstractVector,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ = ADNLPModels.get_lag(nlp, b, obj_weight, y)
  sparse_hess_coord!(ℓ, b, x, vals)
end

function ADNLPModels.hess_coord!(
  b::SparseADHessianSDTColoration,
  nlp::ADNLPModels.ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ = ADNLPModels.get_lag(nlp, b, obj_weight)
  sparse_hess_coord!(ℓ, b, x, vals)
end

function ADNLPModels.hess_coord!(
  b::SparseADHessianSDTColoration,
  nlp::ADNLPModels.ADModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  y = zeros(T, nlp.meta.nnln)
  for (w, k) in enumerate(nlp.meta.nln)
    y[w] = k == j ? 1 : 0
  end
  obj_weight = zero(T)
  ℓ = ADNLPModels.get_lag(nlp, b, obj_weight, y)
  sparse_hess_coord!(ℓ, b, x, vals)
  return vals
end

struct SparseADJacobianSDTColoration{T, Tag, S} <: ADNLPModels.ADBackend
  d::BitVector
  rowval::Vector{Int}
  colptr::Vector{Int}
  colors::Vector{Int}
  ncolors::Int
  z::Vector{ForwardDiff.Dual{Tag, T, 1}}
  cz::Vector{ForwardDiff.Dual{Tag, T, 1}}
  res::S
end

function SparseADJacobianSDTColoration(
  nvar,
  f,
  ncon,
  c!;
  x0::AbstractVector{T} = rand(nvar),
  alg::SparseDiffTools.SparseDiffToolsColoringAlgorithm = SparseDiffTools.GreedyD1Color(),
  kwargs...,
) where {T}
  output = similar(x0, ncon)
  J = Symbolics.jacobian_sparsity(c!, output, x0)
  colors = matrix_colors(J, alg)
  d = BitVector(undef, nvar)
  ncolors = maximum(colors)

  rowval = J.rowval
  colptr = J.colptr

  tag = ForwardDiff.Tag{typeof(c!), T}

  z = Vector{ForwardDiff.Dual{tag, T, 1}}(undef, nvar)
  cz = similar(z, ncon)
  res = similar(x0, ncon)

  SparseADJacobianSDTColoration(d, rowval, colptr, colors, ncolors, z, cz, res)
end

function ADNLPModels.get_nln_nnzj(b::SparseADJacobianSDTColoration, nvar, ncon)
  length(b.rowval)
end

function ADNLPModels.jac_structure!(
  b::SparseADJacobianSDTColoration,
  nlp::ADNLPModels.ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rowval
  for i = 1:(nlp.meta.nvar)
    for j = b.colptr[i]:(b.colptr[i + 1] - 1)
      cols[j] = i
    end
  end
  return rows, cols
end

function ADNLPModels.sparse_jac_coord!(
  ℓ!::Function,
  b::SparseADJacobianSDTColoration{T, Tag},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, Tag}
  nvar = length(x)
  for icol = 1:(b.ncolors)
    b.d .= (b.colors .== icol)
    map!(ForwardDiff.Dual{Tag}, b.z, x, b.d) # x + ε * v
    ℓ!(b.cz, b.z) # c!(cz, x + ε * v)
    ForwardDiff.extract_derivative!(Tag, b.res, b.cz) # ∇c!(cx, x)ᵀv
    for j = 1:nvar
      if b.colors[j] == icol
        for k = b.colptr[j]:(b.colptr[j + 1] - 1)
          i = b.rowval[k]
          vals[k] = b.res[i]
        end
      end
    end
  end
  return vals
end

function ADNLPModels.jac_coord!(b::SparseADJacobianSDTColoration, nlp::ADNLPModels.ADModel, x::AbstractVector, vals::AbstractVector)
  ADNLPModels.sparse_jac_coord!(nlp.c!, b, x, vals)
  return vals
end

function ADNLPModels.jac_structure_residual!(
  b::SparseADJacobianSDTColoration,
  nls::ADNLPModels.AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rowval
  for i = 1:(nls.meta.nvar)
    for j = b.colptr[i]:(b.colptr[i + 1] - 1)
      cols[j] = i
    end
  end
  return rows, cols
end

function ADNLPModels.jac_coord_residual!(
  b::SparseADJacobianSDTColoration,
  nls::ADNLPModels.AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  ADNLPModels.sparse_jac_coord!(nls.F!, b, x, vals)
  return vals
end

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
