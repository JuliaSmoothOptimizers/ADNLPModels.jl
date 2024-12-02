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

struct SparseEnzymeADHessian{R, C, S} <: ADNLPModels.ADBackend
  nvar::Int
  rowval::Vector{Int}
  colptr::Vector{Int}
  nzval::Vector{R}
  result_coloring::C
  coloring_mode::Symbol
  compressed_hessian::S
  v::Vector{R}
  y::Vector{R}
  grad::Vector{R}
end

function SparseEnzymeADHessian(
  nvar,
  f,
  ncon,
  c!;
  x0::AbstractVector = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:substitution}(),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  kwargs...,
)
  H = compute_hessian_sparsity(f, nvar, c!, ncon, detector = detector)
  SparseEnzymeADHessian(nvar, f, ncon, c!, H; x0, coloring_algorithm, kwargs...)
end

function SparseEnzymeADHessian(
  nvar,
  f,
  ncon,
  c!,
  H::SparseMatrixCSC{Bool, Int};
  x0::AbstractVector{T} = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:substitution}(),
  kwargs...,
) where {T}
  problem = ColoringProblem{:symmetric, :column}()
  result_coloring = coloring(H, problem, coloring_algorithm, decompression_eltype = T)

  trilH = tril(H)
  rowval = trilH.rowval
  colptr = trilH.colptr
  nzval = T.(trilH.nzval)
  if coloring_algorithm isa GreedyColoringAlgorithm{:direct}
    coloring_mode = :direct
    compressed_hessian = similar(x0)
  else
    coloring_mode = :substitution
    group = column_groups(result_coloring)
    ncolors = length(group)
    compressed_hessian = similar(x0, (nvar, ncolors))
  end
  v = similar(x0)
  y = similar(x0, ncon)
  grad = similar(x0)

  return SparseEnzymeADHessian(
    nvar,
    rowval,
    colptr,
    nzval,
    result_coloring,
    coloring_mode,
    compressed_hessian,
    v,
    y,
    grad,
  )
end

@init begin
  @require Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9" begin

    function ADNLPModels.gradient(::EnzymeReverseADGradient, f, x)
      g = similar(x)
      Enzyme.gradient!(Enzyme.Reverse, g, Enzyme.Const(f), x)
      return g
    end

    function ADNLPModels.gradient!(::EnzymeReverseADGradient, g, f, x)
      Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(f), Enzyme.Active, Enzyme.Duplicated(x, g))
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
        Enzyme.hvp!(tmp, Enzyme.Const(f), x, seed)
        hess[:, i] .= tmp
        seed[i] = zero(eltype(seed))
      end
      return hess
    end

    function Jprod!(b::EnzymeReverseADJprod, Jv, c!, x, v, ::Val)
      Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(c!), Enzyme.Duplicated(b.x, Jv), Enzyme.Duplicated(x, v))
      return Jv
    end

    function Jtprod!(b::EnzymeReverseADJtprod, Jtv, c!, x, v, ::Val)
      Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(c!), Enzyme.Duplicated(b.x, Jtv), Enzyme.Duplicated(x, v))
      return Jtv
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
        Enzyme.Forward,
        Enzyme.Const(Enzyme.gradient!),
        Enzyme.Const(Enzyme.Reverse),
        Enzyme.DuplicatedNoNeed(b.grad, Hv),
        Enzyme.Const(ℓ),
        Enzyme.Duplicated(x, v),
        Enzyme.Const(y),
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
        Enzyme.Forward,
        Enzyme.Const(Enzyme.gradient!),
        Enzyme.Const(Enzyme.Reverse),
        Enzyme.DuplicatedNoNeed(b.grad, Hv),
        Enzyme.Const(f),
        Enzyme.Duplicated(x, v),
      )
      return Hv
    end

    # Sparse Jacobian
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
        Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(c!), Enzyme.Duplicated(b.buffer, b.compressed_jacobian), Enzyme.Duplicated(x, b.v))

        # Update the columns of the Jacobian that have the color `icol`
        decompress_single_color!(A, b.compressed_jacobian, icol, b.result_coloring)
      end
      vals .= b.nzval
      return vals
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

    # Sparse Hessian
    function get_nln_nnzh(b::SparseEnzymeADHessian, nvar)
      return length(b.rowval)
    end

    function NLPModels.hess_structure!(
      b::SparseEnzymeADHessian,
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

    function sparse_hess_coord!(
      b::SparseEnzymeADHessian,
      x::AbstractVector,
      obj_weight,
      y::AbstractVector,
      vals::AbstractVector,
    )
      # SparseMatrixColorings.jl requires a SparseMatrixCSC for the decompression
      A = SparseMatrixCSC(b.nvar, b.nvar, b.colptr, b.rowval, b.nzval)

      groups = column_groups(b.result_coloring)
      for (icol, cols) in enumerate(groups)
        # Update the seed
        b.v .= 0
        for col in cols
          b.v[col] = 1
        end

        # column icol of the compressed hessian
        compressed_hessian_icol =
          (b.coloring_mode == :direct) ? b.compressed_hessian : view(b.compressed_hessian, :, icol)

        # Lagrangian
        ℓ = get_lag(nlp, b, obj_weight, y)

        # AD with Enzyme.jl
        Enzyme.autodiff(
          Enzyme.Forward,
          Enzyme.Const(Enzyme.gradient!),
          Enzyme.Const(Enzyme.Reverse),
          Enzyme.DuplicatedNoNeed(b.grad, compressed_hessian_icol),
          Enzyme.Const(ℓ),
          Enzyme.Duplicated(x, b.v),
          Enzyme.Const(y),
        )

        if b.coloring_mode == :direct
          # Update the coefficients of the lower triangular part of the Hessian that are related to the color `icol`
          decompress_single_color!(A, compressed_hessian_icol, icol, b.result_coloring, :L)
        end
      end
      if b.coloring_mode == :substitution
        decompress!(A, b.compressed_hessian, b.result_coloring, :L)
      end
      vals .= b.nzval
      return vals
    end

    function NLPModels.hess_coord!(
      b::SparseEnzymeADHessian,
      nlp::ADModel,
      x::AbstractVector,
      y::AbstractVector,
      obj_weight::Real,
      vals::AbstractVector,
    )
      sparse_hess_coord!(b, x, obj_weight, y, vals)
    end

    # Could be optimized!
    function NLPModels.hess_coord!(
      b::SparseEnzymeADHessian,
      nlp::ADModel,
      x::AbstractVector,
      obj_weight::Real,
      vals::AbstractVector,
    )
      b.y .= 0
      sparse_hess_coord!(b, x, obj_weight, b.y, vals)
    end

    function NLPModels.hess_coord!(
      b::SparseEnzymeADHessian,
      nlp::ADModel,
      x::AbstractVector,
      j::Integer,
      vals::AbstractVector,
    )
      for (w, k) in enumerate(nlp.meta.nln)
        b.y[w] = k == j ? 1 : 0
      end
      obj_weight = zero(eltype(x))
      sparse_hess_coord!(b, x, obj_weight, b.y, vals)
      return vals
    end

    function NLPModels.hess_structure_residual!(
      b::SparseEnzymeADHessian,
      nls::AbstractADNLSModel,
      rows::AbstractVector{<:Integer},
      cols::AbstractVector{<:Integer},
    )
      return hess_structure!(b, nls, rows, cols)
    end

    function NLPModels.hess_coord_residual!(
      b::SparseEnzymeADHessian,
      nls::AbstractADNLSModel,
      x::AbstractVector,
      v::AbstractVector,
      vals::AbstractVector,
    )
      obj_weight = zero(eltype(x))
      sparse_hess_coord!(b, x, obj_weight, v, vals)
    end
  end
end
