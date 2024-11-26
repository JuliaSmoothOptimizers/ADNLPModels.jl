struct EnzymeReverseADJacobian <: ADBackend end
struct EnzymeReverseADHessian <: ADBackend end

struct EnzymeReverseADGradient <: ADNLPModels.ADBackend end

function EnzymeReverseADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector = rand(nvar),
  kwargs...,
)
  return EnzymeReverseADGradient()
end

function ADNLPModels.gradient!(::EnzymeReverseADGradient, g, f, x)
  Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Duplicated(x, g)) # gradient!(Reverse, g, f, x)
  return g
end

function EnzymeReverseADJacobian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return EnzymeReverseADJacobian()
end

jacobian(::EnzymeReverseADJacobian, f, x) = Enzyme.jacobian(Enzyme.Reverse, f, x)

function EnzymeReverseADHessian(
  nvar::Integer,

  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzh = nvar * (nvar + 1) / 2
  return EnzymeReverseADHessian()
end

function hessian(::EnzymeReverseADHessian, f, x)
  seed = similar(x)
  hess = zeros(eltype(x), length(x), length(x))
  fill!(seed, zero(x))
  for i in 1:length(x)
    seed[i] = one(x)
    Enzyme.hvp!(view(hess, i, :), f, x, seed)
    seed[i] = zero(x)
  end
  return hess
end

struct EnzymeReverseADJprod <: InPlaceADbackend
  x::Vector{Float64}
end

function EnzymeReverseADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  x = zeros(nvar)
  return EnzymeReverseADJprod(x)
end

function Jprod!(b::EnzymeReverseADJprod, Jv, c!, x, v, ::Val)
  Enzyme.autodiff(Enzyme.Forward, c!, Duplicated(b.x, Jv), Enzyme.Duplicated(x, v))
  return Jv
end

struct EnzymeReverseADJtprod <: InPlaceADbackend
  x::Vector{Float64}
end

function EnzymeReverseADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  x = zeros(nvar)
  return EnzymeReverseADJtprod(x)
end

function Jtvprod!(b::EnzymeReverseADJtprod, Jtv, c!, x, v, ::Val)
  Enzyme.autodiff(Enzyme.Reverse, c!, Duplicated(b.x, Jtv), Enzyme.Duplicated(x, v))
  return Jtv
end

struct EnzymeReverseADHvprod <: InPlaceADbackend
  grad::Vector{Float64}
end

function EnzymeReverseADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  grad = zeros(nvar)
  return EnzymeReverseADHvprod(grad)
end

function Hvprod!(b::EnzymeReverseADHvprod, Hv, x, v, f, args...)
  # What to do with args?
  Enzyme.autodiff(
    Forward,
    gradient!,
    Const(Reverse),
    DuplicatedNoNeed(b.grad, Hv),
    Const(f),
    Duplicated(x, v),
  )
  return Hv
end

function Hvprod!(
  b::EnzymeReverseADHvprod,
  Hv,
  x,
  v,
  ℓ,
  ::Val{:lag},
  y,
  obj_weight::Real = one(eltype(x)),
)
  Enzyme.autodiff(
    Forward,
    gradient!,
    Const(Reverse),
    DuplicatedNoNeed(b.grad, Hv),
    Const(ℓ),
    Duplicated(x, v),
    Const(y),
  )

  return Hv
end

function Hvprod!(
  b::EnzymeReverseADHvprod,
  Hv,
  x,
  v,
  f,
  ::Val{:obj},
  obj_weight::Real = one(eltype(x)),
)
  Enzyme.autodiff(
    Forward,
    gradient!,
    Const(Reverse),
    DuplicatedNoNeed(b.grad, Hv),
    Const(f),
    Duplicated(x, v),
    Const(y),
  )
  return Hv
end
