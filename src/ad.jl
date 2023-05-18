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
