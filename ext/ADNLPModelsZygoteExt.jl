module ADNLPModelsZygoteExt

using Zygote, ADNLPModels
import ADNLPModels: ADModel, AbstractADNLSModel, ADBackend, ImmutableADbackend

struct ZygoteADGradient <: ADBackend end
struct ZygoteADJacobian <: ImmutableADbackend
  nnzj::Int
end
struct ZygoteADHessian <: ImmutableADbackend
  nnzh::Int
end
struct ZygoteADJprod <: ImmutableADbackend end
struct ZygoteADJtprod <: ImmutableADbackend end
# See https://fluxml.ai/Zygote.jl/latest/limitations/
function get_immutable_c(nlp::ADModel)
  function c(x; nnln = nlp.meta.nnln)
    c = Zygote.Buffer(x, nnln)
    nlp.c!(c, x)
    return copy(c)
  end
  return c
end
get_c(nlp::ADModel, ::ImmutableADbackend) = get_immutable_c(nlp)

function get_immutable_F(nls::AbstractADNLSModel)
  function F(x; nequ = nls.nls_meta.nequ)
    Fx = Zygote.Buffer(x, nequ)
    nls.F!(Fx, x)
    return copy(Fx)
  end
  return F
end
get_F(nls::AbstractADNLSModel, ::ImmutableADbackend) = get_immutable_F(nls)

function ZygoteADGradient(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ZygoteADGradient()
end
function gradient(::ZygoteADGradient, f, x)
  g = Zygote.gradient(f, x)[1]
  return g === nothing ? zero(x) : g
end
function gradient!(::ZygoteADGradient, g, f, x)
  _g = Zygote.gradient(f, x)[1]
  g .= _g === nothing ? 0 : _g
end

function ZygoteADJacobian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzj = nvar * ncon
  return ZygoteADJacobian(nnzj)
end
function jacobian(::ZygoteADJacobian, f, x)
  return Zygote.jacobian(f, x)[1]
end

function ZygoteADHessian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  @assert nvar > 0
  nnzh = nvar * (nvar + 1) / 2
  return ZygoteADHessian(nnzh)
end
function hessian(b::ZygoteADHessian, f, x)
  return jacobian(
    ForwardDiffADJacobian(length(x), f, x0 = x),
    x -> gradient(ZygoteADGradient(), f, x),
    x,
  )
end

function ZygoteADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ZygoteADJprod()
end
function Jprod!(::ZygoteADJprod, Jv, f, x, v, ::Val)
  Jv .= vec(Zygote.jacobian(t -> f(x + t * v), 0)[1])
  return Jv
end

function ZygoteADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return ZygoteADJtprod()
end
function Jtprod!(::ZygoteADJtprod, Jtv, f, x, v, ::Val)
  g = Zygote.gradient(x -> dot(f(x), v), x)[1]
  if g === nothing
    Jtv .= zero(x)
  else
    Jtv .= g
  end
  return Jtv
end

end