struct GenericForwardDiffADGradient <: ADBackend end
GenericForwardDiffADGradient(args...; kwargs...) = GenericForwardDiffADGradient()
function gradient!(::GenericForwardDiffADGradient, g, f, x)
  return ForwardDiff.gradient!(g, f, x)
end

struct ForwardDiffADGradient <: ADBackend
  cfg
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
function Jprod!(::GenericForwardDiffADJprod, Jv, f, x, v)
  Jv .= ForwardDiff.derivative(t -> f(x + t * v), 0)
  return Jv
end

struct ForwardDiffADJprod{T, Tag} <: ADNLPModels.InPlaceADbackend
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

function ADNLPModels.Jprod!(b::ForwardDiffADJprod{T, Tag}, Jv, c!, x, v) where {T, Tag}
  map!(ForwardDiff.Dual{Tag}, b.z, x, v) # x + ε * v
  c!(b.cz, b.z) # c!(cz, x + ε * v)
  ForwardDiff.extract_derivative!(Tag, Jv, b.cz) # ∇c!(cx, x)ᵀv
  return Jv
end

struct ForwardDiffADJtprod <: ADBackend end
function ForwardDiffADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ForwardDiffADJtprod()
end
function Jtprod!(::ForwardDiffADJtprod, Jtv, f, x, v)
  Jtv .= ForwardDiff.gradient(x -> dot(f(x), v), x)
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
function Hvprod!(::GenericForwardDiffADHvprod, Hv, f, x, v)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(f, x + t * v), 0)
  return Hv
end

struct ForwardDiffADHvprod{T} <: ADBackend
  tmp_in::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
  tmp_out::Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}
end
function ForwardDiffADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  tmp_in =
    Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  tmp_out =
    Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, nvar)
  return ForwardDiffADHvprod(tmp_in, tmp_out)
end

function Hvprod!(b::ForwardDiffADHvprod, Hv, f, x, v)
  ϕ!(dy, x; f = f) = ForwardDiff.gradient!(dy, f, x)
  SparseDiffTools.auto_hesvecgrad!(Hv, (dy, x) -> ϕ!(dy, x), x, v, b.tmp_in, b.tmp_out)
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
