include("additional_hprod_backends.jl")

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

using Symbolics

struct SparseForwardADHessian <: ADNLPModels.ADBackend
  d::BitVector
  rowval::Vector{Int}
  colptr::Vector{Int}
  colors::Vector{Int}
  ncolors::Int
end

function SparseForwardADHessian(nvar, f, ncon, c!;
  x0=rand(nvar),
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
    fun = fun + dot(c!(cx,xsi), ysi)
  end
  S = Symbolics.hessian_sparsity(fun, ncon == 0 ? xsi : [xsi; ysi], full=false)
  H = ncon == 0 ? S : S[1:nvar,1:nvar]
  rows, cols, _ = findnz(H)
  colors = matrix_colors(H, alg)
  d = BitVector(undef, nvar)
  ncolors = maximum(colors)
  return SparseForwardADHessian(d, H.rowval, H.colptr, colors, ncolors)
end

function ADNLPModels.get_nln_nnzh(b::SparseForwardADHessian, nvar)
  return length(b.rowval)
end

function ADNLPModels.hess_structure!(
  b::SparseForwardADHessian,
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

function sparse_hess_coord!(
  ℓ::Function,
  b::SparseForwardADHessian,
  x::AbstractVector,
  vals::AbstractVector
  )
  nvar = length(x)
  for icol = 1 : b.ncolors
    b.d .= (b.colors .== icol)
    res = ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * b.d), 0)
    for j = 1 : nvar
      if b.colors[j] == icol
        for k = b.colptr[j] : b.colptr[j+1] - 1
          i = b.rowval[k]
          vals[k] = res[i]
        end
      end
    end
  end
  return vals
end

function ADNLPModels.hess_coord!(
  b::SparseForwardADHessian,
  nlp::ADNLPModels.ADModel,
  x::AbstractVector,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ = ADNLPModels.get_lag(nlp, b, obj_weight, y)
  sparse_hess_coord!(ℓ, b, x, vals)
end

function hess_coord!(
  b::SparseForwardADHessian,
  nlp::ADNLPModels.ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ(x) = obj_weight * nlp.f(x)
  sparse_hess_coord!(ℓ, b, x, vals)
end
