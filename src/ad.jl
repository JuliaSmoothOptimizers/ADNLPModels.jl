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
  gradient_backend = get_default_backend(:gradient_backend, backend),
  hprod_backend = get_default_backend(:hprod_backend, backend),
  hessian_backend = get_default_backend(:hessian_backend, backend, matrix_free),
  kwargs...,
)
  c! = (args...) -> []
  ncon = 0

  GB = gradient_backend
  b = @elapsed begin
    gradient_backend = if gradient_backend isa AbstractNLPModel
      gradient_backend
    else
      GB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("gradient backend $GB: $b seconds;")

  HvB = hprod_backend
  b = @elapsed begin
    hprod_backend = if hprod_backend isa AbstractNLPModel
      hprod_backend
    else
      HvB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("hprod    backend $HvB: $b seconds;")

  HB = hessian_backend
  b = @elapsed begin
    hessian_backend = if hessian_backend isa AbstractNLPModel
      hessian_backend
    else
      HB(nvar, f, ncon, c!; kwargs...)
    end
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
  gradient_backend = get_default_backend(:gradient_backend, backend),
  hprod_backend = get_default_backend(:hprod_backend, backend),
  jprod_backend = get_default_backend(:jprod_backend, backend),
  jtprod_backend = get_default_backend(:jtprod_backend, backend),
  jacobian_backend = get_default_backend(:jacobian_backend, backend, matrix_free),
  hessian_backend = get_default_backend(:hessian_backend, backend, matrix_free),
  ghjvprod_backend = get_default_backend(:ghjvprod_backend, backend),
  kwargs...,
)

  GB = gradient_backend
  b = @elapsed begin
    gradient_backend = if gradient_backend isa AbstractNLPModel
      gradient_backend
    else
      GB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("gradient backend $GB: $b seconds;")

  HvB = hprod_backend
  b = @elapsed begin
    hprod_backend = if hprod_backend isa AbstractNLPModel
      hprod_backend
    else
      HvB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("hprod    backend $HvB: $b seconds;")

  JvB = jprod_backend
  b = @elapsed begin
    jprod_backend = if jprod_backend isa AbstractNLPModel
      jprod_backend
    else
      JvB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("jprod    backend $JvB: $b seconds;")

  JtvB = jtprod_backend
  b = @elapsed begin
    jtprod_backend = if jtprod_backend isa AbstractNLPModel
      jtprod_backend
    else
      JtvB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("jtprod   backend $JtvB: $b seconds;")

  JB = jacobian_backend
  b = @elapsed begin
    jacobian_backend = if jacobian_backend isa AbstractNLPModel
      jacobian_backend
    else
      JB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("jacobian backend $JB: $b seconds;")

  HB = hessian_backend
  b = @elapsed begin
    hessian_backend = if hessian_backend isa AbstractNLPModel
      hessian_backend
    else
      HB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("hessian  backend $HB: $b seconds;")

  GHJ = ghjvprod_backend
  b = @elapsed begin
    ghjvprod_backend = if ghjvprod_backend isa AbstractNLPModel
      ghjvprod_backend
    else
      GHJ(nvar, f, ncon, c!; kwargs...)
    end
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
  gradient_backend = get_default_backend(:gradient_backend, backend),
  hprod_backend = get_default_backend(:hprod_backend, backend),
  hessian_backend = get_default_backend(:hessian_backend, backend, matrix_free),
  hprod_residual_backend = get_default_backend(:hprod_residual_backend, backend),
  jprod_residual_backend = get_default_backend(:jprod_residual_backend, backend),
  jtprod_residual_backend = get_default_backend(:jtprod_residual_backend, backend),
  jacobian_residual_backend = get_default_backend(:jacobian_residual_backend, backend, matrix_free),
  hessian_residual_backend = get_default_backend(:hessian_residual_backend, backend, matrix_free),
  kwargs...,
)
  function F(x; nequ = nequ)
    Fx = similar(x, nequ)
    F!(Fx, x)
    return Fx
  end
  f = x -> mapreduce(Fi -> Fi^2, +, F(x)) / 2

  c! = (args...) -> []
  ncon = 0

  GB = gradient_backend
  b = @elapsed begin
    gradient_backend = if gradient_backend isa AbstractNLPModel
      gradient_backend
    else
      GB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("gradient backend $GB: $b seconds;")

  HvB = hprod_backend
  b = @elapsed begin
    hprod_backend = if hprod_backend isa AbstractNLPModel
      hprod_backend
    else
      HvB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("hprod    backend $HvB: $b seconds;")

  HB = hessian_backend
  b = @elapsed begin
    hessian_backend = if hessian_backend isa AbstractNLPModel
      hessian_backend
    else
      HB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("hessian  backend $HB: $b seconds;")

  HvBLS = hprod_residual_backend
  b = @elapsed begin
    hprod_residual_backend = if hprod_residual_backend isa AbstractNLPModel
      hprod_residual_backend
    else
      HvBLS(nvar, f, nequ, F!; kwargs...)
    end
  end
  show_time && println("hprod_residual    backend $HvBLS: $b seconds;")

  JvBLS = jprod_residual_backend
  b = @elapsed begin
    jprod_residual_backend = if jprod_residual_backend isa AbstractNLPModel
      jprod_residual_backend
    else
      JvBLS(nvar, f, nequ, F!; kwargs...)
    end
  end
  show_time && println("jprod_residual    backend $JvBLS: $b seconds;")

  JtvBLS = jtprod_residual_backend
  b = @elapsed begin
    jtprod_residual_backend = if jtprod_residual_backend isa AbstractNLPModel
      jtprod_residual_backend
    else
      JtvBLS(nvar, f, nequ, F!; kwargs...)
    end
  end
  show_time && println("jtprod_residual   backend $JtvBLS: $b seconds;")

  JBLS = jacobian_residual_backend
  b = @elapsed begin
    jacobian_residual_backend = if jacobian_residual_backend isa AbstractNLPModel
      jacobian_residual_backend
    else
      JBLS(nvar, f, nequ, F!; kwargs...)
    end
  end
  show_time && println("jacobian_residual backend $JBLS: $b seconds;")

  HBLS = hessian_residual_backend
  b = @elapsed begin
    hessian_residual_backend = if hessian_residual_backend isa AbstractNLPModel
      hessian_residual_backend
    else
      HBLS(nvar, f, nequ, F!; kwargs...)
    end
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
  gradient_backend = get_default_backend(:gradient_backend, backend),
  hprod_backend = get_default_backend(:hprod_backend, backend),
  jprod_backend = get_default_backend(:jprod_backend, backend),
  jtprod_backend = get_default_backend(:jtprod_backend, backend),
  jacobian_backend = get_default_backend(:jacobian_backend, backend, matrix_free),
  hessian_backend = get_default_backend(:hessian_backend, backend, matrix_free),
  ghjvprod_backend = get_default_backend(:ghjvprod_backend, backend),
  hprod_residual_backend = get_default_backend(:hprod_residual_backend, backend),
  jprod_residual_backend = get_default_backend(:jprod_residual_backend, backend),
  jtprod_residual_backend = get_default_backend(:jtprod_residual_backend, backend),
  jacobian_residual_backend = get_default_backend(:jacobian_residual_backend, backend, matrix_free),
  hessian_residual_backend = get_default_backend(:hessian_residual_backend, backend, matrix_free),
  kwargs...,
)
  function F(x; nequ = nequ)
    Fx = similar(x, nequ)
    F!(Fx, x)
    return Fx
  end
  f = x -> mapreduce(Fi -> Fi^2, +, F(x)) / 2

  GB = gradient_backend
  b = @elapsed begin
    gradient_backend = if gradient_backend isa AbstractNLPModel
      gradient_backend
    else
      GB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("gradient backend $GB: $b seconds;")

  HvB = hprod_backend
  b = @elapsed begin
    hprod_backend = if hprod_backend isa AbstractNLPModel
      hprod_backend
    else
      HvB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("hprod    backend $HvB: $b seconds;")

  JvB = jprod_backend
  b = @elapsed begin
    jprod_backend = if jprod_backend isa AbstractNLPModel
      jprod_backend
    else
      JvB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("jprod    backend $JvB: $b seconds;")

  JtvB = jtprod_backend
  b = @elapsed begin
    jtprod_backend = if jtprod_backend isa AbstractNLPModel
      jtprod_backend
    else
      JtvB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("jtprod   backend $JtvB: $b seconds;")

  JB = jacobian_backend
  b = @elapsed begin
    jacobian_backend = if jacobian_backend isa AbstractNLPModel
      jacobian_backend
    else
      JB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("jacobian backend $JB: $b seconds;")

  HB = hessian_backend
  b = @elapsed begin
    hessian_backend = if hessian_backend isa AbstractNLPModel
      hessian_backend
    else
      HB(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("hessian  backend $HB: $b seconds;")

  GHJ = ghjvprod_backend
  b = @elapsed begin
    ghjvprod_backend = if ghjvprod_backend isa AbstractNLPModel
      ghjvprod_backend
    else
      GHJ(nvar, f, ncon, c!; kwargs...)
    end
  end
  show_time && println("ghjvprod backend $GHJ: $b seconds. \n")

  HvBLS = hprod_residual_backend
  b = @elapsed begin
    hprod_residual_backend = if hprod_residual_backend isa AbstractNLPModel
      hprod_residual_backend
    else
      HvBLS(nvar, f, nequ, F!; kwargs...)
    end
  end
  show_time && println("hprod_residual    backend $HvBLS: $b seconds;")

  JvBLS = jprod_residual_backend
  b = @elapsed begin
    jprod_residual_backend = if jprod_residual_backend isa AbstractNLPModel
      jprod_residual_backend
    else
      JvBLS(nvar, f, nequ, F!; kwargs...)
    end
  end
  show_time && println("jprod_residual    backend $JvBLS: $b seconds;")

  JtvBLS = jtprod_residual_backend
  b = @elapsed begin
    jtprod_residual_backend = if jtprod_residual_backend isa AbstractNLPModel
      jtprod_residual_backend
    else
      JtvBLS(nvar, f, nequ, F!; kwargs...)
    end
  end
  show_time && println("jtprod_residual   backend $JtvBLS: $b seconds;")

  JBLS = jacobian_residual_backend
  b = @elapsed begin
    jacobian_residual_backend = if jacobian_residual_backend isa AbstractNLPModel
      jacobian_residual_backend
    else
      JBLS(nvar, f, nequ, F!; kwargs...)
    end
  end
  show_time && println("jacobian_residual backend $JBLS: $b seconds;")

  HBLS = hessian_residual_backend
  b = @elapsed begin
    hessian_residual_backend = if hessian_residual_backend isa AbstractNLPModel
      hessian_residual_backend
    else
      HBLS(nvar, f, nequ, F!; kwargs...)
    end
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
