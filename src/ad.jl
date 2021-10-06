abstract type GradientADBackend end

struct GradientForwardDiff{T} <: GradientADBackend
  cfg::T
end
struct GradientZygote <: GradientADBackend end
struct GradientReverseDiff{T} <: GradientADBackend
  cfg::T
end

gradient!(b::GradientADBackend, ::Any, ::Any, ::Any) = throw_error(b)
function gradient!(adbackend::GradientForwardDiff, g, f, x)
  return ForwardDiff.gradient!(g, f, x, adbackend.cfg)
end

@init begin
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    function gradient!(::GradientZygote, g, f, x)
      _g = Zygote.gradient(f, x)[1]
      g .= _g === nothing ? 0 : _g
    end
  end
end
@init begin
  @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
    function gradient!(adbackend::GradientReverseDiff, g, f, x)
      return ReverseDiff.gradient!(g, adbackend.cfg, x)
    end
  end
end

abstract type HvADBackend end # use in-place gradient

struct HvForwardDiff <: HvADBackend
  _g
end
struct HvZygote <: HvADBackend end
struct HvReverseDiff <: HvADBackend
  _g
end

hvprod!(b::HvADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)
function hvprod!(b::HvForwardDiff, f, x, v, Hv)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(f, x + t * v), 0)  # use in-place derivative!
  return Hv
end
@init begin
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    function hvprod!(b::HvZygote, f, x, v, Hv)
      Hv .= ForwardDiff.derivative(t -> Zygote.gradient(f, x + t * v), 0)
      return Hv
    end
  end
end
@init begin
  @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
    function hvprod!(b::HvReverseDiff, f, x, v, Hv)
      Hv .= ForwardDiff.derivative(t -> ReverseDiff.gradient(f, x + t * v), 0)  # use in-place derivative!
      return Hv
    end
  end
end

abstract type JvADBackend end # use in-place c function

struct JvForwardDiff <: JvADBackend end
struct JvZygote <: JvADBackend end
struct JvReverseDiff <: JvADBackend end

jprod!(b::JvADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)
function jprod!(::JvForwardDiff, c, x, v, Jv)
  Jv .= ForwardDiff.derivative(t -> c(x + t * v), 0) # use in-place derivative!
  return Jv
end
@init begin
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    function jprod!(::JvZygote, c, x, v, Jv)
      Jv .= vec(Zygote.jacobian(t -> c(x + t * v), 0)[1])
      return Jv
    end
  end
end
@init begin
  @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
    function jprod!(::JvReverseDiff, c, x, v, Jv)
      T = eltype(x)
      return ReverseDiff.jacobian!(Jv, t -> c(x + t[1] * v), [T(0)])
    end
  end
end

abstract type JtvADBackend end # use in-place c function

struct JtvForwardDiff <: JtvADBackend end
struct JtvZygote <: JtvADBackend end
struct JtvReverseDiff <: JtvADBackend end

jtprod!(b::JtvADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)
function jtprod!(::JtvForwardDiff, c, x, v, Jtv)
  return ForwardDiff.gradient!(Jtv, x -> dot(c(x), v), x)
end
@init begin
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    function jtprod!(::JtvZygote, c, x, v, Jtv)
      g = Zygote.gradient(x -> dot(c(x), v), x)[1]
      if g === nothing
        return zero(x)
      else
        Jtv .= g
      end
      return Jtv
    end
  end
end
@init begin
  @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
    function jtprod!(::JtvReverseDiff, c, x, v, Jtv)
      return ReverseDiff.gradient!(Jtv, x -> dot(c(x), v), x)
    end
  end
end

abstract type JacobianADBackend end

struct JacobianForwardDiff <: JacobianADBackend end
struct JacobianZygote <: JacobianADBackend end
struct JacobianReverseDiff <: JacobianADBackend end

function jac_structure!(
  b::JacobianADBackend,
  nlp,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  m, n = nlp.meta.ncon, nlp.meta.nvar
  I = ((i, j) for i = 1:m, j = 1:n)
  rows .= getindex.(I, 1)[:]
  cols .= getindex.(I, 2)[:]
  return rows, cols
end
function jac_coord!(b::JacobianADBackend, nlp, x::AbstractVector, vals::AbstractVector)
  Jx = jacobian(b, nlp.c, x)
  vals .= Jx[:]
  return vals
end

jacobian(b::JacobianADBackend, ::Any, ::Any) = throw_error(b)
jacobian(::JacobianForwardDiff, c, x) = ForwardDiff.jacobian(c, x)
@init begin
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    jacobian(::JacobianZygote, c, x) = Zygote.jacobian(c, x)[1]
  end
end
@init begin
  @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
    jacobian(::JacobianReverseDiff, c, x) = ReverseDiff.jacobian(c, x)
  end
end

abstract type HessianADBackend end

struct HessianForwardDiff <: HessianADBackend end
struct HessianZygote <: HessianADBackend end
struct HessianReverseDiff <: HessianADBackend end

function hess_structure!(
  b::HessianADBackend,
  nlp,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  n = nlp.meta.nvar
  I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end
function hess_coord!(b::HessianADBackend, nlp, x::AbstractVector, ℓ::Function, vals::AbstractVector)
  Hx = hessian(b, ℓ, x)
  k = 1
  for j = 1:(nlp.meta.nvar)
    for i = j:(nlp.meta.nvar)
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

hessian(b::HessianADBackend, ::Any, ::Any) = throw_error(b)
hessian(::HessianForwardDiff, f, x) = ForwardDiff.hessian(f, x)
@init begin
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    hessian(::HessianZygote, f, x) = Zygote.hessian(f, x)
  end
end
@init begin
  @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
    hessian(::HessianReverseDiff, f, x) = ReverseDiff.hessian(f, x)
  end
end

# TODO:
function directional_second_derivative(::HessianADBackend, f, x, v, w)
  return ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> f(x + s * w + t * v), 0), 0)
end

export ADModelBackend

abstract type AbstractADModelBackend end

struct ADModelBackend{
  GB <: GradientADBackend,
  HvB <: HvADBackend,
  JvB <: JvADBackend,
  JtvB <: JtvADBackend,
  JB <: JacobianADBackend,
  HB <: HessianADBackend,
} <: AbstractADModelBackend

  nnzh::Int
  nnzj::Int

  gradient_backend::GB
  hprod_backend::HvB
  jprod_backend::JvB
  jtprod_backend::JtvB
  jacobian_backend::JB
  hessian_backend::HB
end

ADModelBackend(n, f, x0) = ADModelBackend(n, 0, f, x -> eltype(x0)[], x0)

function ADModelBackend(
  nvar::Integer,
  ncon::Integer,
  f,
  c,
  x0::AbstractVector;
  gradient_backend::GB = GradientForwardDiff(ForwardDiff.GradientConfig(f, x0)),
  hprod_backend::HvB = HvForwardDiff(similar(x0, nvar)),
  jprod_backend::JvB = JvForwardDiff(),
  jtprod_backend::JtvB = JtvForwardDiff(),
  jacobian_backend::JB = JacobianForwardDiff(),
  hessian_backend::HB = HessianForwardDiff(),
) where {GB, HvB, JvB, JtvB, JB, HB}
  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon
  
  return ADModelBackend{GB, HvB, JvB, JtvB, JB, HB}(
    nnzh,
    nnzj,
    gradient_backend,
    hprod_backend,
    jprod_backend,
    jtprod_backend,
    jacobian_backend,
    hessian_backend,
  )
end
