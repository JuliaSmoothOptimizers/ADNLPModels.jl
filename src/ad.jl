"""
    ADModelBackend(gradient_backend, hprod_backend, jprod_backend, jtprod_backend, jacobian_backend, hessian_backend, ghjvprod_backend, hprod_residual_backend, jprod_residual_backend, jtprod_residual_backend, jacobian_residual_backend, hessian_residual_backend)

Structure that define the different backend used to compute automatic differentiation of an `ADNLPModel`/`ADNLSModel` model.
The different backend are all subtype of `ADBackend` and are respectively used for:
  - gradient computation;
  - hessian-vector products;
  - jacobian-vector products;
  - transpose jacobian-vector products;
  - jacobian computation;
  - hessian computation;
  - directional second derivative computation, i.e. gᵀ ∇²cᵢ(x) v.

The default constructors are 
    ADModelBackend(nvar, f, ncon = 0, c::Function = (args...) -> []; show_time::Bool = false, kwargs...)
    ADModelNLSBackend(nvar, F!, nequ, ncon = 0, c::Function = (args...) -> []; show_time::Bool = false, kwargs...)

If `show_time` is set to `true`, it prints the time used to generate each backend.

The remaining `kwargs` are either the different backends as listed below or arguments passed to the backend's constructors:
  - `gradient_backend = ForwardDiffADGradient`;
  - `hprod_backend = ForwardDiffADHvprod`;
  - `jprod_backend = ForwardDiffADJprod`;
  - `jtprod_backend = ForwardDiffADJtprod`;
  - `jacobian_backend = SparseADJacobian`;
  - `hessian_backend = ForwardDiffADHessian`;
  - `ghjvprod_backend = ForwardDiffADGHjvprod`;
  - `hprod_residual_backend = ForwardDiffADHvprod` for `ADNLSModel` and `EmptyADbackend` otherwise;
  - `jprod_residual_backend = ForwardDiffADJprod` for `ADNLSModel` and `EmptyADbackend` otherwise;
  - `jtprod_residual_backend = ForwardDiffADJtprod` for `ADNLSModel` and `EmptyADbackend` otherwise;
  - `jacobian_residual_backend = SparseADJacobian` for `ADNLSModel` and `EmptyADbackend` otherwise;
  - `hessian_residual_backend = ForwardDiffADHessian` for `ADNLSModel` and `EmptyADbackend` otherwise.

"""
struct ADModelBackend{GB, HvB, JvB, JtvB, JB, HB, GHJ, HvBLS, JvBLS, JtvBLS, JBLS, HBLS}
  gradient_backend::GB
  hprod_backend::HvB
  jprod_backend::JvB
  jtprod_backend::JtvB
  jacobian_backend::JB
  hessian_backend::HB
  ghjvprod_backend::GHJ

  hprod_residual_backend::HvBLS
  jprod_residual_backend::JvBLS
  jtprod_residual_backend::JtvBLS
  jacobian_residual_backend::JBLS
  hessian_residual_backend::HBLS
end

function Base.show(
  io::IO,
  backend::ADModelBackend{GB, HvB, JvB, JtvB, JB, HB, GHJ},
) where {GB, HvB, JvB, JtvB, JB, HB, GHJ}
  print(io, replace(replace(
    "ADModelBackend{
  $GB,
  $HvB,
  $JvB,
  $JtvB,
  $JB,
  $HB,
  $GHJ,
}",
    "ADNLPModels." => "",
  ), r"\{(.+)\}" => s""))
end

function ADModelBackend(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  show_time::Bool = false,
  gradient_backend::Type{GB} = ForwardDiffADGradient,
  hprod_backend::Type{HvB} = ForwardDiffADHvprod,
  jprod_backend::Type{JvB} = ForwardDiffADJprod,
  jtprod_backend::Type{JtvB} = ForwardDiffADJtprod,
  jacobian_backend::Type{JB} = SparseADJacobian,
  hessian_backend::Type{HB} = SparseADHessian,
  ghjvprod_backend::Type{GHJ} = ForwardDiffADGHjvprod,
  kwargs...,
) where {GB, HvB, JvB, JtvB, JB, HB, GHJ}
  b = @elapsed begin
    gradient_backend = GB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("gradient backend $GB: $b seconds;")
  b = @elapsed begin
    hprod_backend = HvB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("hprod    backend $HvB: $b seconds;")
  b = @elapsed begin
    jprod_backend = JvB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("jprod    backend $JvB: $b seconds;")
    b = @elapsed begin
    jtprod_backend = JtvB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("jtprod   backend $JtvB: $b seconds;")
    b = @elapsed begin
    jacobian_backend = JB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("jacobian backend $JB: $b seconds;")
    b = @elapsed begin
    hessian_backend = HB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("hessian  backend $HB: $b seconds;")
    b = @elapsed begin
    ghjvprod_backend = GHJ(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("ghjvprod backend $GHJ: $b seconds. \n")
  return ADModelBackend(
    gradient_backend,
    hprod_backend,
    jprod_backend,
    jtprod_backend,
    jacobian_backend,
    hessian_backend,
    ghjvprod_backend,
    EmptyADbackend(),
    EmptyADbackend(),
    EmptyADbackend(),
    EmptyADbackend(),
    EmptyADbackend(),
  )
end

function ADModelNLSBackend(
  nvar::Integer,
  F!,
  nequ::Integer,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  show_time::Bool = false,
  gradient_backend::Type{GB} = ForwardDiffADGradient,
  hprod_backend::Type{HvB} = ForwardDiffADHvprod,
  jprod_backend::Type{JvB} = ForwardDiffADJprod,
  jtprod_backend::Type{JtvB} = ForwardDiffADJtprod,
  jacobian_backend::Type{JB} = SparseADJacobian,
  hessian_backend::Type{HB} = SparseADHessian,
  ghjvprod_backend::Type{GHJ} = ForwardDiffADGHjvprod,
  hprod_residual_backend::Type{HvBLS} = ForwardDiffADHvprod,
  jprod_residual_backend::Type{JvBLS} = ForwardDiffADJprod,
  jtprod_residual_backend::Type{JtvBLS} = ForwardDiffADJtprod,
  jacobian_residual_backend::Type{JBLS} = SparseADJacobian,
  hessian_residual_backend::Type{HBLS} = ForwardDiffADHessian,
  kwargs...,
) where {GB, HvB, JvB, JtvB, JB, HB, GHJ, HvBLS, JvBLS, JtvBLS, JBLS, HBLS}
  function F(x; nequ = nequ)
    Fx = similar(x, nequ)
    F!(Fx, x)
    return Fx
  end
  f = x -> mapreduce(Fi -> Fi^2, +, F(x)) / 2

  b = @elapsed begin
    gradient_backend = GB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("gradient          backend $GB: $b seconds;")
  b = @elapsed begin
    hprod_backend = HvB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("hprod             backend $HvB: $b seconds;")
  b = @elapsed begin
    jprod_backend = JvB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("jprod             backend $JvB: $b seconds;")
    b = @elapsed begin
    jtprod_backend = JtvB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("jtprod            backend $JtvB: $b seconds;")
    b = @elapsed begin
    jacobian_backend = JB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("jacobian          backend $JB: $b seconds;")
    b = @elapsed begin
    hessian_backend = HB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("hessian           backend $HB: $b seconds;")
    b = @elapsed begin
    ghjvprod_backend = GHJ(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("ghjvprod          backend $GHJ: $b seconds;")

  b = @elapsed begin
    hprod_residual_backend = HvBLS(nvar, f, nequ, F!; kwargs...)
  end
  show_time && println("hprod_residual    backend $HvBLS: $b seconds;")
  b = @elapsed begin
    jprod_residual_backend = JvBLS(nvar, f, nequ, F!; kwargs...)
  end
  show_time && println("jprod_residual    backend $JvBLS: $b seconds;")
  b = @elapsed begin
    jtprod_residual_backend = JtvBLS(nvar, f, nequ, F!; kwargs...)
  end
  show_time && println("jtprod_residual   backend $JtvBLS: $b seconds;")
  b = @elapsed begin
    jacobian_residual_backend = JBLS(nvar, f, nequ, F!; kwargs...)
  end
  show_time && println("jacobian_residual backend $JBLS: $b seconds;")
  b = @elapsed begin
    hessian_residual_backend = HBLS(nvar, f, nequ, F!; kwargs...)
  end
  show_time && println("hessian_residual  backend $HBLS: $b seconds. \n")

  return ADModelBackend(
    gradient_backend,
    hprod_backend,
    jprod_backend,
    jtprod_backend,
    jacobian_backend,
    hessian_backend,
    ghjvprod_backend,
    hprod_residual_backend,
    jprod_residual_backend,
    jtprod_residual_backend,
    jacobian_residual_backend,
    hessian_residual_backend,
  )
end

abstract type ADBackend end
abstract type ImmutableADbackend <: ADBackend end
abstract type InPlaceADbackend <: ADBackend end

struct EmptyADbackend <: ADBackend end

"""
    get_nln_nnzj(::ADBackend, nvar, ncon)
    get_nln_nnzj(b::ADModelBackend, nvar, ncon)

For a given `ADBackend` of a problem with `nvar` variables and `ncon` constraints, return the number of nonzeros in the Jacobian of nonlinear constraints.
If `b` is the `ADModelBackend` then `b.jacobian_backend` is used.
"""
function get_nln_nnzj(b::ADModelBackend, nvar, ncon)
  get_nln_nnzj(b.jacobian_backend, nvar, ncon)
end

function get_nln_nnzj(::ADBackend, nvar, ncon)
  nvar * ncon
end

"""
    get_residual_nnzj(b::ADModelBackend, nvar, nequ)

Return `get_nln_nnzj(b.jacobian_residual_backend, nvar, nequ)`.
"""
function get_residual_nnzj(b::ADModelBackend, nvar, nequ)
  get_nln_nnzj(b.jacobian_residual_backend, nvar, nequ)
end

"""
    get_nln_nnzh(::ADBackend, nvar)
    get_nln_nnzh(b::ADModelBackend, nvar)

For a given `ADBackend` of a problem with `nvar` variables, return the number of nonzeros in the lower triangle of the Hessian.
If `b` is the `ADModelBackend` then `b.hessian_backend` is used.
"""
function get_nln_nnzh(b::ADModelBackend, nvar)
  get_nln_nnzh(b.hessian_backend, nvar)
end

function get_nln_nnzh(::ADBackend, nvar)
  div(nvar * (nvar + 1), 2)
end

throw_error(b) =
  throw(ArgumentError("The AD backend $b is not loaded. Please load the corresponding AD package."))
gradient(b::ADBackend, ::Any, ::Any) = throw_error(b)
gradient!(b::ADBackend, ::Any, ::Any, ::Any) = throw_error(b)
jacobian(b::ADBackend, ::Any, ::Any) = throw_error(b)
hessian(b::ADBackend, ::Any, ::Any) = throw_error(b)
Jprod!(b::ADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)
Jtprod!(b::ADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)
Hvprod!(b::ADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)
directional_second_derivative(::ADBackend, ::Any, ::Any, ::Any, ::Any) = throw_error(b)

function hess_structure!(
  b::ADBackend,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  n = nlp.meta.nvar
  pos = 0
  for j = 1:n
    for i = j:n
      pos += 1
      rows[pos] = i
      cols[pos] = j
    end
  end
  return rows, cols
end

function hess_coord!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ = get_lag(nlp, b, obj_weight, y)
  hess_coord!(b, nlp, x, ℓ, vals)
  return vals
end

function hess_coord!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ = get_lag(nlp, b, obj_weight)
  return hess_coord!(b, nlp, x, ℓ, vals)
end

function hess_coord!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector,
)
  c = get_c(nlp, b)
  ℓ = x -> c(x)[j - nlp.meta.nlin]
  Hx = hessian(b, ℓ, x)
  k = 1
  n = nlp.meta.nvar
  for j = 1:n
    for i = j:n
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hess_coord!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  ℓ::Function,
  vals::AbstractVector,
)
  Hx = hessian(b, ℓ, x)
  k = 1
  n = nlp.meta.nvar
  for j = 1:n
    for i = j:n
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hprod!(
  b::ADBackend,
  nlp::ADModel,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  Hv::AbstractVector,
)
  c = get_c(nlp, b)
  Hvprod!(b, Hv, x -> c(x)[j - nlp.meta.nlin], x, v)
  return Hv
end

function jac_structure!(
  b::ADBackend,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  m, n = nlp.meta.nnln, nlp.meta.nvar
  return jac_dense!(m, n, rows, cols)
end

function jac_structure_residual!(
  b::ADBackend,
  nls::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  return jac_dense!(m, n, rows, cols)
end

function jac_dense!(
  m::Integer,
  n::Integer,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  pos = 0
  for j = 1:n
    for i = 1:m
      pos += 1
      rows[pos] = i
      cols[pos] = j
    end
  end
  return rows, cols
end

function jac_coord!(b::ADBackend, nlp::ADModel, x::AbstractVector, vals::AbstractVector)
  c = get_c(nlp, b)
  Jx = jacobian(b, c, x)
  vals .= view(Jx, :)
  return vals
end

function jac_coord_residual!(
  b::ADBackend,
  nls::AbstractADNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  F = get_F(nls, b)
  Jx = jacobian(b, F, x)
  vals .= view(Jx, :)
  return vals
end
