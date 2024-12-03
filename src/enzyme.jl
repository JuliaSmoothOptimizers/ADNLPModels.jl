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

struct EnzymeReverseADHessian{T} <: ADBackend
  seed::Vector{T}
  Hv::Vector{T}
end

function EnzymeReverseADHessian(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  @assert nvar > 0
  nnzh = nvar * (nvar + 1) / 2

  seed = zeros(T, nvar)
  Hv = zeros(T, nvar)
  return EnzymeReverseADHessian(seed, Hv)
end

struct EnzymeReverseADHvprod{T} <: InPlaceADbackend
  grad::Vector{T}
end

function EnzymeReverseADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  grad = zeros(T, nvar)
  return EnzymeReverseADHvprod(grad)
end

struct EnzymeReverseADJprod{T} <: InPlaceADbackend
  cx::Vector{T}
end

function EnzymeReverseADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  cx = zeros(T, nvar)
  return EnzymeReverseADJprod(cx)
end

struct EnzymeReverseADJtprod{T} <: InPlaceADbackend
  cx::Vector{T}
end

function EnzymeReverseADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  cx = zeros(T, nvar)
  return EnzymeReverseADJtprod(cx)
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
  cx::Vector{R}
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
  cx = zeros(T, ncon)

  SparseEnzymeADJacobian(
    nvar,
    ncon,
    rowval,
    colptr,
    nzval,
    result_coloring,
    compressed_jacobian,
    v,
    cx,
  )
end

struct SparseEnzymeADHessian{R, C, S, L} <: ADNLPModels.ADBackend
  nvar::Int
  rowval::Vector{Int}
  colptr::Vector{Int}
  nzval::Vector{R}
  result_coloring::C
  coloring_mode::Symbol
  compressed_hessian_icol::Vector{R}
  compressed_hessian::S
  v::Vector{R}
  y::Vector{R}
  grad::Vector{R}
  cx::Vector{R}
  ℓ::L
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
    compressed_hessian_icol = similar(x0)
    compressed_hessian = compressed_hessian_icol
  else
    coloring_mode = :substitution
    group = column_groups(result_coloring)
    ncolors = length(group)
    compressed_hessian_icol = similar(x0)
    compressed_hessian = similar(x0, (nvar, ncolors))
  end
  v = similar(x0)
  y = similar(x0, ncon)
  cx = similar(x0, ncon)
  grad = similar(x0)
  function ℓ(x, y, obj_weight, cx)
    res = obj_weight * f(x)
    if ncon != 0
      c!(cx, x)
      res += sum(cx[i] * y[i] for i = 1:ncon)
    end
    return res
  end

  return SparseEnzymeADHessian(
    nvar,
    rowval,
    colptr,
    nzval,
    result_coloring,
    coloring_mode,
    compressed_hessian_icol,
    compressed_hessian,
    v,
    y,
    grad,
    cx,
    ℓ,
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

    function hessian(b::EnzymeReverseADHessian, f, x)
      T = eltype(x)
      n = length(x)
      hess = zeros(T, n, n)
      fill!(b.seed, zero(T))
      for i in 1:n
        b.seed[i] = one(T)
        Enzyme.hvp!(b.Hv, Enzyme.Const(f), x, b.seed)
        view(hess, :, i) .= b.Hv
        b.seed[i] = zero(T)
      end
      return hess
    end

    function Jprod!(b::EnzymeReverseADJprod, Jv, c!, x, v, ::Val)
      Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(c!), Enzyme.Duplicated(b.cx, Jv), Enzyme.Duplicated(x, v))
      return Jv
    end

    function Jtprod!(b::EnzymeReverseADJtprod, Jtv, c!, x, v, ::Val)
      Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(c!), Enzyme.Duplicated(b.cx, Jtv), Enzyme.Duplicated(x, v))
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
        Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(c!), Enzyme.Duplicated(b.cx, b.compressed_jacobian), Enzyme.Duplicated(x, b.v))

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

        function _gradient!(dx, f, x, y, obj_weight, cx)
          Enzyme.make_zero!(dx)
          res = Enzyme.autodiff(
            Enzyme.Reverse,
            f,
            Enzyme.Active,
            Enzyme.Duplicated(x, dx),
            Enzyme.Const(y),
            Enzyme.Const(obj_weight),
            Enzyme.Const(cx)
          )
          return nothing
        end

        function _hvp!(res, f, x, v, y, obj_weight, cx)
          # grad = Enzyme.make_zero(x)
          Enzyme.autodiff(
              Enzyme.Forward,
              _gradient!,
              res,
              Enzyme.Const(f),
              Enzyme.Duplicated(x, v),
              Enzyme.Const(y),
              Enzyme.Const(obj_weight),
              Enzyme.Const(cx),
          )
          return nothing
        end

        _hvp!(
          Enzyme.DuplicatedNoNeed(b.grad, b.compressed_hessian_icol),
          b.ℓ, x, b.v, y, obj_weight, b.cx
        )

        if b.coloring_mode == :direct
          # Update the coefficients of the lower triangular part of the Hessian that are related to the color `icol`
          decompress_single_color!(A, b.compressed_hessian_icol, icol, b.result_coloring, :L)
        end
        if b.coloring_mode == :substitution
          view(b.compressed_hessian, :, icol) .= b.compressed_hessian_icol
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
