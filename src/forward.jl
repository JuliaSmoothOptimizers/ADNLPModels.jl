struct GenericForwardDiffADGradient <: ADBackend end
GenericForwardDiffADGradient(args...; kwargs...) = GenericForwardDiffADGradient()
function gradient!(::GenericForwardDiffADGradient, g, f, x)
  return ForwardDiff.gradient!(g, f, x)
end

struct ForwardDiffADGradient{GC} <: ADBackend
  cfg::GC
end
function ForwardDiffADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  @assert nvar > 0
  @lencheck nvar x0
  cfg = ForwardDiff.GradientConfig(f, x0)
  return ForwardDiffADGradient(cfg)
end
function gradient!(adbackend::ForwardDiffADGradient, g, f, x)
  return ForwardDiff.gradient!(g, f, x, adbackend.cfg)
end

struct ForwardDiffADJacobian <: ADBackend
  nnzj::Int
end
function ForwardDiffADJacobian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzj = nvar * ncon
  return ForwardDiffADJacobian(nnzj)
end
jacobian(::ForwardDiffADJacobian, f, x) = ForwardDiff.jacobian(f, x)

struct ForwardDiffADHessian <: ADBackend
  nnzh::Int
end
function ForwardDiffADHessian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzh = nvar * (nvar + 1) / 2
  return ForwardDiffADHessian(nnzh)
end
hessian(::ForwardDiffADHessian, f, x) = ForwardDiff.hessian(f, x)

struct GenericForwardDiffADJprod <: ADBackend end
function GenericForwardDiffADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericForwardDiffADJprod()
end
function Jprod!(::GenericForwardDiffADJprod, Jv, f, x, v, ::Val)
  Jv .= ForwardDiff.derivative(t -> f(x + t * v), 0)
  return Jv
end

struct ForwardDiffADJprod{T, Tag} <: InPlaceADbackend
  z::Vector{ForwardDiff.Dual{Tag, T, 1}}
  cz::Vector{ForwardDiff.Dual{Tag, T, 1}}
end

function ForwardDiffADJprod(
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
  return ForwardDiffADJprod(z, cz)
end

function Jprod!(b::ForwardDiffADJprod{T, Tag}, Jv, c!, x, v, ::Val) where {T, Tag}
  map!(ForwardDiff.Dual{Tag}, b.z, x, v) # x + ε * v
  c!(b.cz, b.z) # c!(cz, x + ε * v)
  ForwardDiff.extract_derivative!(Tag, Jv, b.cz) # ∇c!(cx, x)ᵀv
  return Jv
end

struct GenericForwardDiffADJtprod <: ADBackend end
function GenericForwardDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericForwardDiffADJtprod()
end
function Jtprod!(::GenericForwardDiffADJtprod, Jtv, f, x, v, ::Val)
  Jtv .= ForwardDiff.gradient(x -> dot(f(x), v), x)
  return Jtv
end

struct ForwardDiffADJtprod{Tag, GT, S} <: InPlaceADbackend
  cfg::ForwardDiff.GradientConfig{Tag}
  ψ::GT
  temp::S
  sol::S
end

function ForwardDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  temp = similar(x0, nvar + 2 * ncon)
  sol = similar(x0, nvar + 2 * ncon)

  function ψ(z; nvar = nvar, ncon = ncon)
    cx, x, u = view(z, 1:ncon),
    view(z, (ncon + 1):(nvar + ncon)),
    view(z, (nvar + ncon + 1):(nvar + ncon + ncon))
    c!(cx, x)
    dot(cx, u)
  end
  tagψ = ForwardDiff.Tag(ψ, T)
  cfg = ForwardDiff.GradientConfig(ψ, temp, ForwardDiff.Chunk(temp), tagψ)

  return ForwardDiffADJtprod(cfg, ψ, temp, sol)
end

function Jtprod!(b::ForwardDiffADJtprod{Tag, GT, S}, Jtv, c!, x, v, ::Val) where {Tag, GT, S}
  ncon = length(v)
  nvar = length(x)

  b.sol[1:ncon] .= 0
  b.sol[(ncon + 1):(ncon + nvar)] .= x
  b.sol[(ncon + nvar + 1):(2 * ncon + nvar)] .= v
  ForwardDiff.gradient!(b.temp, b.ψ, b.sol, b.cfg)
  Jtv .= view(b.temp, (ncon + 1):(nvar + ncon))
  return Jtv
end

struct GenericForwardDiffADHvprod <: ADBackend end
function GenericForwardDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return GenericForwardDiffADHvprod()
end
function Hvprod!(::GenericForwardDiffADHvprod, Hv, x, v, f, args...)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(f, x + t * v), 0)
  return Hv
end

struct ForwardDiffADHvprod{Tag, GT, S, T, F, Tagf} <: ADBackend
  lz::Vector{ForwardDiff.Dual{Tag, T, 1}}
  glz::Vector{ForwardDiff.Dual{Tag, T, 1}}
  sol::S
  longv
  Hvp
  ∇φ!::GT
  z::Vector{ForwardDiff.Dual{Tagf, T, 1}}
  gz::Vector{ForwardDiff.Dual{Tagf, T, 1}}
  ∇f!::F
end

function ForwardDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::S = rand(nvar),
  kwargs...,
) where {S}
  T = eltype(S)
  function lag(z; nvar = nvar, ncon = ncon, f = f, c! = c!)
    cx, x, y, ob = view(z, 1:ncon),
    view(z, (ncon + 1):(nvar + ncon)),
    view(z, (nvar + ncon + 1):(nvar + ncon + ncon)),
    z[end]
    if ncon > 0
      c!(cx, x)
      return ob * f(x) + dot(cx, y)
    else
      return ob * f(x)
    end
  end

  ntotal = nvar + 2 * ncon + 1

  sol = similar(x0, ntotal)
  lz = Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(lag), T}, T, 1}}(undef, ntotal)
  glz = similar(lz)
  cfg = ForwardDiff.GradientConfig(lag, lz)
  function ∇φ!(gz, z; lag = lag, cfg = cfg)
    ForwardDiff.gradient!(gz, lag, z, cfg)
    return gz
  end
  longv = fill!(S(undef, ntotal), 0)
  Hvp = fill!(S(undef, ntotal), 0)

  # unconstrained Hessian
  tagf = ForwardDiff.Tag{typeof(f), T}
  z = Vector{ForwardDiff.Dual{tagf, T, 1}}(undef, nvar)
  gz = similar(z)
  cfgf = ForwardDiff.GradientConfig(f, z)
  ∇f!(gz, z; f = f, cfgf = cfgf) = ForwardDiff.gradient!(gz, f, z, cfgf)

  return ForwardDiffADHvprod(lz, glz, sol, longv, Hvp, ∇φ!, z, gz, ∇f!)
end

function Hvprod!(
  b::ForwardDiffADHvprod{Tag, GT, S, T},
  Hv,
  x::AbstractVector{T},
  v,
  ℓ,
  ::Val{:lag},
  y,
  obj_weight::Real = one(T),
) where {Tag, GT, S, T}
  nvar = length(x)
  ncon = Int((length(b.sol) - nvar - 1) / 2)
  b.sol[1:ncon] .= zero(T)
  b.sol[(ncon + 1):(ncon + nvar)] .= x
  b.sol[(ncon + nvar + 1):(2 * ncon + nvar)] .= y
  b.sol[end] = obj_weight

  b.longv .= 0
  b.longv[(ncon + 1):(ncon + nvar)] .= v
  map!(ForwardDiff.Dual{Tag}, b.lz, b.sol, b.longv)

  b.∇φ!(b.glz, b.lz)
  ForwardDiff.extract_derivative!(Tag, b.Hvp, b.glz)
  Hv .= view(b.Hvp, (ncon + 1):(ncon + nvar))
  return Hv
end

function Hvprod!(
  b::ForwardDiffADHvprod{Tag, GT, S, T, F, Tagf},
  Hv,
  x::AbstractVector{T},
  v,
  f,
  ::Val{:obj},
  obj_weight::Real = one(T),
) where {Tag, GT, S, T, F, Tagf}
  map!(ForwardDiff.Dual{Tagf}, b.z, x, v) # x + ε * v
  b.∇f!(b.gz, b.z) # ∇f(x + ε * v) = ∇f(x) + ε * ∇²f(x)ᵀv
  ForwardDiff.extract_derivative!(Tagf, Hv, b.gz)  # ∇²f(x)ᵀv
  Hv .*= obj_weight
  return Hv
end

function NLPModels.hprod!(
  b::ForwardDiffADHvprod{Tag, GT, S, T},
  nlp::ADModel,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
) where {Tag, GT, S, T}
  nvar = nlp.meta.nvar
  ncon = nlp.meta.nnln

  b.sol[1:ncon] .= 0
  b.sol[(ncon + 1):(ncon + nvar)] .= x
  k = 0
  for i = 1:(nlp.meta.ncon)
    if i in nlp.meta.nln
      k += 1
      b.sol[ncon + nvar + k] = i == j ? one(T) : zero(T)
    end
  end

  b.sol[end] = zero(T)

  b.longv .= 0
  b.longv[(ncon + 1):(ncon + nvar)] .= v
  map!(ForwardDiff.Dual{Tag}, b.lz, b.sol, b.longv)

  b.∇φ!(b.glz, b.lz)
  ForwardDiff.extract_derivative!(Tag, b.Hvp, b.glz)
  Hv .= view(b.Hvp, (ncon + 1):(ncon + nvar))
  return Hv
end

function NLPModels.hprod_residual!(
  b::ForwardDiffADHvprod{Tag, GT, S, T},
  nls::AbstractADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
) where {Tag, GT, S, T}
  nvar = nls.meta.nvar
  nequ = nls.nls_meta.nequ

  b.sol[1:nequ] .= 0
  b.sol[(nequ + 1):(nequ + nvar)] .= x
  for i = 1:nequ
    b.sol[nequ + nvar + i] = i == j ? one(T) : zero(T)
  end

  b.sol[end] = zero(T)

  b.longv .= 0
  b.longv[(nequ + 1):(nequ + nvar)] .= v

  map!(ForwardDiff.Dual{Tag}, b.lz, b.sol, b.longv)

  b.∇φ!(b.glz, b.lz)

  ForwardDiff.extract_derivative!(Tag, b.Hvp, b.glz)
  Hv .= view(b.Hvp, (nequ + 1):(nequ + nvar))
  return Hv
end

struct ForwardDiffADGHjvprod <: ADBackend end
function ForwardDiffADGHjvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ForwardDiffADGHjvprod()
end
function directional_second_derivative(::ForwardDiffADGHjvprod, f, x, v, w)
  return ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> f(x + s * w + t * v), 0), 0)
end
