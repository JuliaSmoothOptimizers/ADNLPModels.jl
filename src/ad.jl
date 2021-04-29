abstract type ADBackend end
struct ForwardDiffAD <: ADBackend
  nnzh::Int
  nnzj::Int
end
function ForwardDiffAD(f, c, x0::AbstractVector, ncon::Integer)
  nvar = length(x0)
  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon
  return ForwardDiffAD(nnzh, nnzj)
end
function ForwardDiffAD(f, x0::AbstractVector)
  nvar = length(x0)
  nnzh = nvar * (nvar + 1) / 2
  return ForwardDiffAD(nnzh, 0)
end
struct ZygoteAD <: ADBackend
  nnzh::Int
  nnzj::Int
end
function ZygoteAD(f, c, x0::AbstractVector, ncon::Integer)
  nvar = length(x0)
  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon
  return ZygoteAD(nnzh, nnzj)
end
function ZygoteAD(f, x0::AbstractVector)
  nvar = length(x0)
  nnzh = nvar * (nvar + 1) / 2
  return ZygoteAD(nnzh, 0)
end
struct ReverseDiffAD <: ADBackend
  nnzh::Int
  nnzj::Int
end
function ReverseDiffAD(f, c, x0::AbstractVector, ncon::Integer)
  nvar = length(x0)
  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon
  return ReverseDiffAD(nnzh, nnzj)
end
function ReverseDiffAD(f, x0::AbstractVector)
  nvar = length(x0)
  nnzh = nvar * (nvar + 1) / 2
  return ReverseDiffAD(nnzh, 0)
end

throw_error(b) =
  throw(ArgumentError("The AD backend $b is not loaded. Please load the corresponding AD package."))
gradient(b::ADBackend, ::Any, ::Any) = throw_error(b)
gradient!(b::ADBackend, ::Any, ::Any, ::Any) = throw_error(b)
jacobian(b::ADBackend, ::Any, ::Any) = throw_error(b)
hessian(b::ADBackend, ::Any, ::Any) = throw_error(b)
Jprod(b::ADBackend, ::Any, ::Any, ::Any) = throw_error(b)
Jtprod(b::ADBackend, ::Any, ::Any, ::Any) = throw_error(b)
function hess_structure!(
  b::ADBackend,
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
function hess_coord!(b::ADBackend, nlp, x::AbstractVector, ℓ::Function, vals::AbstractVector)
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
function jac_structure!(
  b::ADBackend,
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
function jac_coord!(b::ADBackend, nlp, x::AbstractVector, vals::AbstractVector)
  Jx = jacobian(b, nlp.c, x)
  vals .= Jx[:]
  return vals
end
function directional_second_derivative(::ADBackend, f, x, v, w)
  return ForwardDiff.derivative(t -> ForwardDiff.derivative(s -> f(x + s * w + t * v), 0), 0)
end
function Hvprod(b::ADBackend, f, x, v)
  return ForwardDiff.derivative(t -> gradient(b, f, x + t * v), 0)
end

gradient(::ForwardDiffAD, f, x) = ForwardDiff.gradient(f, x)
function gradient!(::ForwardDiffAD, g, f, x)
  return ForwardDiff.gradient!(g, f, x)
end
jacobian(::ForwardDiffAD, f, x) = ForwardDiff.jacobian(f, x)
hessian(::ForwardDiffAD, f, x) = ForwardDiff.hessian(f, x)
function Jprod(::ForwardDiffAD, f, x, v)
  return ForwardDiff.derivative(t -> f(x + t * v), 0)
end
function Jtprod(::ForwardDiffAD, f, x, v)
  return ForwardDiff.gradient(x -> dot(f(x), v), x)
end

@init begin
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    function gradient(::ZygoteAD, f, x)
      g = Zygote.gradient(f, x)[1]
      return g === nothing ? zero(x) : g
    end
    function gradient!(::ZygoteAD, g, f, x)
      _g = Zygote.gradient(f, x)[1]
      g .= _g === nothing ? 0 : _g
    end
    function jacobian(::ZygoteAD, f, x)
      return Zygote.jacobian(f, x)[1]
    end
    function hessian(b::ZygoteAD, f, x)
      return jacobian(ForwardDiffAD(f, x), x -> gradient(b, f, x), x)
    end
    function Jprod(::ZygoteAD, f, x, v)
      return vec(Zygote.jacobian(t -> f(x + t * v), 0)[1])
    end
    function Jtprod(::ZygoteAD, f, x, v)
      g = Zygote.gradient(x -> dot(f(x), v), x)[1]
      return g === nothing ? zero(x) : g
    end
  end
  @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
    gradient(::ReverseDiffAD, f, x) = ReverseDiff.gradient(f, x)
    function gradient!(::ReverseDiffAD, g, f, x)
      return ReverseDiff.gradient!(g, f, x)
    end
    jacobian(::ReverseDiffAD, f, x) = ReverseDiff.jacobian(f, x)
    hessian(::ReverseDiffAD, f, x) = ReverseDiff.hessian(f, x)
    function Jprod(::ReverseDiffAD, f, x, v)
      return vec(ReverseDiff.jacobian(t -> f(x + t[1] * v), [0.0]))
    end
    function Jtprod(::ReverseDiffAD, f, x, v)
      return ReverseDiff.gradient(x -> dot(f(x), v), x)
    end
  end
end
