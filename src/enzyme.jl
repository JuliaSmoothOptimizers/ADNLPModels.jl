struct EnzymeReverseADGradient <: InPlaceADbackend end

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

struct EnzymeReverseADJacobian <: ADBackend end

function EnzymeReverseADJacobian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  kwargs...,
)
  return EnzymeReverseADJacobian()
end

struct EnzymeReverseADHessian <: ADBackend end

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

struct SparseEnzymeADJacobian{R, C, S} <: ADBackend
  nvar::Int
  ncon::Int
  rowval::Vector{Int}
  colptr::Vector{Int}
  nzval::Vector{R}
  result_coloring::C
  compressed_jacobian::S
  v::Vector{R}
  buffer::Vector{R}
end

function SparseEnzymeADJacobian(
  nvar,
  f,
  ncon,
  c!;
  x0::AbstractVector = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:direct}(),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  kwargs...,
)
  output = similar(x0, ncon)
  J = compute_jacobian_sparsity(c!, output, x0, detector = detector)
  SparseEnzymeADJacobian(nvar, f, ncon, c!, J; x0, coloring_algorithm, kwargs...)
end

function SparseEnzymeADJacobian(
  nvar,
  f,
  ncon,
  c!,
  J::SparseMatrixCSC{Bool, Int};
  x0::AbstractVector{T} = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:direct}(),
  kwargs...,
) where {T}
  # We should support :row and :bidirectional in the future
  problem = ColoringProblem{:nonsymmetric, :column}()
  result_coloring = coloring(J, problem, coloring_algorithm, decompression_eltype = T)

  rowval = J.rowval
  colptr = J.colptr
  nzval = T.(J.nzval)
  compressed_jacobian = similar(x0, ncon)
  v = similar(x0)
  buffer = zeros(T, ncon)

  SparseEnzymeADJacobian(
    nvar,
    ncon,
    rowval,
    colptr,
    nzval,
    result_coloring,
    compressed_jacobian,
    v,
    buffer,
  )
end

@init begin
  @require Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9" begin

  function ADNLPModels.gradient(::EnzymeReverseADGradient, f, x)
    g = similar(x)
    # Enzyme.autodiff(Enzyme.Reverse, Const(f), Active, Enzyme.Duplicated(x, g)) # gradient!(Reverse, g, f, x)
    Enzyme.gradient!(Reverse, g, Const(f), x)
    return g
  end

  function ADNLPModels.gradient!(::EnzymeReverseADGradient, g, f, x)
    Enzyme.autodiff(Enzyme.Reverse, Const(f), Active, Enzyme.Duplicated(x, g)) # gradient!(Reverse, g, f, x)
    return g
  end

  jacobian(::EnzymeReverseADJacobian, f, x) = Enzyme.jacobian(Enzyme.Reverse, f, x)

  function hessian(::EnzymeReverseADHessian, f, x)
    seed = similar(x)
    hess = zeros(eltype(x), length(x), length(x))
    fill!(seed, zero(eltype(x)))
    tmp = similar(x)
    for i in 1:length(x)
      seed[i] = one(eltype(seed))
      Enzyme.hvp!(tmp, Const(f), x, seed)
      hess[:, i] .= tmp
      seed[i] = zero(eltype(seed))
    end
    return hess
  end

  function Jprod!(b::EnzymeReverseADJprod, Jv, c!, x, v, ::Val)
    Enzyme.autodiff(Enzyme.Forward, Const(c!), Duplicated(b.x, Jv), Duplicated(x, v))
    return Jv
  end

  function Jtprod!(b::EnzymeReverseADJtprod, Jtv, c!, x, v, ::Val)
    Enzyme.autodiff(Enzyme.Reverse, Const(c!), Duplicated(b.x, Jtv), Enzyme.Duplicated(x, v))
    return Jtv
  end

  function Hvprod!(b::EnzymeReverseADHvprod, Hv, x, v, f, args...)
    # What to do with args?
    Enzyme.autodiff(
      Forward,
      Const(Enzyme.gradient!),
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
      Const(Enzyme.gradient!),
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
      Const(Enzyme.gradient!),
      Const(Reverse),
      DuplicatedNoNeed(b.grad, Hv),
      Const(f),
      Duplicated(x, v),
      Const(y),
    )
    return Hv
  end

  # Sparse Jacobian
  function sparse_jac_coord!(
    c!::Function,
    b::SparseEnzymeADJacobian,
    x::AbstractVector,
    vals::AbstractVector,
  )
    # SparseMatrixColorings.jl requires a SparseMatrixCSC for the decompression
    A = SparseMatrixCSC(b.ncon, b.nvar, b.colptr, b.rowval, b.nzval)

    groups = column_groups(b.result_coloring)
    for (icol, cols) in enumerate(groups)
      # Update the seed
      b.v .= 0
      for col in cols
        b.v[col] = 1
      end

      # b.compressed_jacobian is just a vector Jv here
      # We don't use the vector mode
      Enzyme.autodiff(Enzyme.Forward, Const(c!), Duplicated(b.buffer, b.compressed_jacobian), Duplicated(x, b.v))

      # Update the columns of the Jacobian that have the color `icol`
      decompress_single_color!(A, b.compressed_jacobian, icol, b.result_coloring)
    end
    vals .= b.nzval
    return vals
  end

  function get_nln_nnzj(b::SparseEnzymeADJacobian, nvar, ncon)
    length(b.rowval)
  end

  function NLPModels.jac_structure!(
    b::SparseEnzymeADJacobian,
    nlp::ADModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
  )
    rows .= b.rowval
    for i = 1:(nlp.meta.nvar)
      for j = b.colptr[i]:(b.colptr[i + 1] - 1)
        cols[j] = i
      end
    end
    return rows, cols
  end

  function NLPModels.jac_coord!(
    b::SparseEnzymeADJacobian,
    nlp::ADModel,
    x::AbstractVector,
    vals::AbstractVector,
  )
    sparse_jac_coord!(nlp.c!, b, x, vals)
    return vals
  end

  function NLPModels.jac_structure_residual!(
    b::SparseEnzymeADJacobian,
    nls::AbstractADNLSModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
  )
    rows .= b.rowval
    for i = 1:(nls.meta.nvar)
      for j = b.colptr[i]:(b.colptr[i + 1] - 1)
        cols[j] = i
      end
    end
    return rows, cols
  end

  function NLPModels.jac_coord_residual!(
    b::SparseEnzymeADJacobian,
    nls::AbstractADNLSModel,
    x::AbstractVector,
    vals::AbstractVector,
  )
    sparse_jac_coord!(nls.F!, b, x, vals)
    return vals
  end
end

