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
    ADModelBackend(nvar, f, ncon = 0, c = (args...) -> []; show_time::Bool = false, kwargs...)
    ADModelNLSBackend(nvar, F!, nequ, ncon = 0, c = (args...) -> []; show_time::Bool = false, kwargs...)

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
  f;
  backend::Symbol = :default,
  matrix_free::Bool = false,
  show_time::Bool = false,
  gradient_backend::Type{GB} = get_default_backend(:gradient_backend, backend),
  hprod_backend::Type{HvB} = get_default_backend(:hprod_backend, backend),
  hessian_backend::Type{HB} = get_default_backend(:hessian_backend, backend, matrix_free),
  kwargs...,
) where {GB, HvB, HB}
  c! = (args...) -> []
  ncon = 0

  b = @elapsed begin
    gradient_backend = GB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("gradient backend $GB: $b seconds;")
  b = @elapsed begin
    hprod_backend = HvB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("hprod    backend $HvB: $b seconds;")
  b = @elapsed begin
    hessian_backend = HB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("hessian  backend $HB: $b seconds;")

  return ADModelBackend(
    gradient_backend,
    hprod_backend,
    EmptyADbackend(),
    EmptyADbackend(),
    EmptyADbackend(),
    hessian_backend,
    EmptyADbackend(),
    EmptyADbackend(),
    EmptyADbackend(),
    EmptyADbackend(),
    EmptyADbackend(),
    EmptyADbackend(),
  )
end

function ADModelBackend(
  nvar::Integer,
  f,
  ncon::Integer,
  c!;
  backend::Symbol = :default,
  matrix_free::Bool = false,
  show_time::Bool = false,
  gradient_backend::Type{GB} = get_default_backend(:gradient_backend, backend),
  hprod_backend::Type{HvB} = get_default_backend(:hprod_backend, backend),
  jprod_backend::Type{JvB} = get_default_backend(:jprod_backend, backend),
  jtprod_backend::Type{JtvB} = get_default_backend(:jtprod_backend, backend),
  jacobian_backend::Type{JB} = get_default_backend(:jacobian_backend, backend, matrix_free),
  hessian_backend::Type{HB} = get_default_backend(:hessian_backend, backend, matrix_free),
  ghjvprod_backend::Type{GHJ} = get_default_backend(:ghjvprod_backend, backend),
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
  nequ::Integer;
  backend::Symbol = :default,
  matrix_free::Bool = false,
  show_time::Bool = false,
  gradient_backend::Type{GB} = get_default_backend(:gradient_backend, backend),
  hprod_backend::Type{HvB} = get_default_backend(:hprod_backend, backend),
  hessian_backend::Type{HB} = get_default_backend(:hessian_backend, backend, matrix_free),
  ghjvprod_backend::Type{GHJ} = get_default_backend(:ghjvprod_backend, backend),
  hprod_residual_backend::Type{HvBLS} = get_default_backend(:hprod_residual_backend, backend),
  jprod_residual_backend::Type{JvBLS} = get_default_backend(:jprod_residual_backend, backend),
  jtprod_residual_backend::Type{JtvBLS} = get_default_backend(:jtprod_residual_backend, backend),
  jacobian_residual_backend::Type{JBLS} = get_default_backend(:jacobian_residual_backend, backend, matrix_free),
  hessian_residual_backend::Type{HBLS} = get_default_backend(:hessian_residual_backend, backend, matrix_free),
  kwargs...,
) where {GB, HvB, HB, GHJ, HvBLS, JvBLS, JtvBLS, JBLS, HBLS}
  function F(x; nequ = nequ)
    Fx = similar(x, nequ)
    F!(Fx, x)
    return Fx
  end
  f = x -> mapreduce(Fi -> Fi^2, +, F(x)) / 2

  c! = (args...) -> []
  ncon = 0

  b = @elapsed begin
    gradient_backend = GB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("gradient          backend $GB: $b seconds;")
  b = @elapsed begin
    hprod_backend = HvB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("hprod             backend $HvB: $b seconds;")
  b = @elapsed begin
    hessian_backend = HB(nvar, f, ncon, c!; kwargs...)
  end
  show_time && println("hessian           backend $HB: $b seconds;")

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
    EmptyADbackend(),
    EmptyADbackend(),
    EmptyADbackend(),
    hessian_backend,
    EmptyADbackend(),
    hprod_residual_backend,
    jprod_residual_backend,
    jtprod_residual_backend,
    jacobian_residual_backend,
    hessian_residual_backend,
  )
end

function ADModelNLSBackend(
  nvar::Integer,
  F!,
  nequ::Integer,
  ncon::Integer,
  c!;
  backend::Symbol = :default,
  matrix_free::Bool = false,
  show_time::Bool = false,
  gradient_backend::Type{GB} = get_default_backend(:gradient_backend, backend),
  hprod_backend::Type{HvB} = get_default_backend(:hprod_backend, backend),
  jprod_backend::Type{JvB} = get_default_backend(:jprod_backend, backend),
  jtprod_backend::Type{JtvB} = get_default_backend(:jtprod_backend, backend),
  jacobian_backend::Type{JB} = get_default_backend(:jacobian_backend, backend, matrix_free),
  hessian_backend::Type{HB} = get_default_backend(:hessian_backend, backend, matrix_free),
  ghjvprod_backend::Type{GHJ} = get_default_backend(:ghjvprod_backend, backend),
  hprod_residual_backend::Type{HvBLS} = get_default_backend(:hprod_residual_backend, backend),
  jprod_residual_backend::Type{JvBLS} = get_default_backend(:jprod_residual_backend, backend),
  jtprod_residual_backend::Type{JtvBLS} = get_default_backend(:jtprod_residual_backend, backend),
  jacobian_residual_backend::Type{JBLS} = get_default_backend(:jacobian_residual_backend, backend, matrix_free),
  hessian_residual_backend::Type{HBLS} = get_default_backend(:hessian_residual_backend, backend, matrix_free),
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

abstract type ColorationAlgorithm end

struct ColPackColoration{F, C, O} <: ColorationAlgorithm
  partition_choice::F
  coloring::C
  ordering::O
end

function ColPackColoration(;
  partition_choice = (m, n) -> false, # TODO: (m, n; μ = 0.6) -> n < μ * m ? true : false,
  coloring::ColPack.AbstractColoring = d1_coloring("DISTANCE_ONE"),
  ordering::ColPack.AbstractOrdering = incidence_degree_ordering("INCIDENCE_DEGREE"),
)
  return ColPackColoration{typeof(partition_choice), typeof(coloring), typeof(ordering)}(
    partition_choice,
    coloring,
    ordering,
  )
end

function sparse_matrix_colors(A, alg::ColPackColoration)
  m, n = size(A)
  partition_by_rows = alg.partition_choice(m, n)
  if !isempty(A.nzval)
    adjA = ColPack.matrix2adjmatrix(A; partition_by_rows = partition_by_rows)
    CPC = ColPackColoring(adjA, alg.coloring, alg.ordering)
    colors = get_colors(CPC)
  else
    colors = zeros(Int, n)
  end
  return colors
end
