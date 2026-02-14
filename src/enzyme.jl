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

struct EnzymeReverseADHessian{T, F} <: ADBackend
  seed::Vector{T}
  Hv::Vector{T}
  f::F
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
  return EnzymeReverseADHessian(seed, Hv, f)
end

struct EnzymeReverseADHvprod{V, F, C, L} <: InPlaceADbackend
  grad::V     # length nvar, gradient buffer (primal in DuplicatedNoNeed)
  hvbuf::V    # length nvar, Hv output buffer (shadow in DuplicatedNoNeed)
  xbuf::V     # length nvar, input x buffer
  vbuf::V     # length nvar, input v buffer (tangent direction)
  cx::V       # length ncon, constraint output buffer
  ybuf::V     # length ncon, multiplier buffer for jth_hprod
  f::F
  c!::C
  ℓ::L
  ncon::Int
end

function EnzymeReverseADHvprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c!::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  grad = fill!(similar(x0), zero(T))
  hvbuf = similar(x0)
  xbuf = similar(x0)
  vbuf = similar(x0)
  cx = fill!(similar(x0, ncon), zero(T))
  ybuf = fill!(similar(x0, ncon), zero(T))

  function ℓ(x, y, obj_weight, cx)
    if ncon != 0
      c!(cx, x)
    end
    res = obj_weight * f(x)
    if ncon != 0
      for i = 1:ncon
        res += cx[i] * y[i]
      end
    end
    return res
  end

  return EnzymeReverseADHvprod(grad, hvbuf, xbuf, vbuf, cx, ybuf, f, c!, ℓ, ncon)
end

struct EnzymeReverseADJprod{V} <: InPlaceADbackend
  cx::V       # length ncon, primal output buffer
  xbuf::V     # length nvar, input x buffer
  vbuf::V     # length nvar, input v buffer (tangent direction)
  jvbuf::V    # length ncon, output Jv buffer
end

function EnzymeReverseADJprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  cx = fill!(similar(x0, ncon), zero(T))
  xbuf = similar(x0)
  vbuf = similar(x0)
  jvbuf = fill!(similar(x0, ncon), zero(T))
  return EnzymeReverseADJprod(cx, xbuf, vbuf, jvbuf)
end

struct EnzymeReverseADJtprod{V} <: InPlaceADbackend
  cx::V       # length ncon, primal output buffer
  xbuf::V     # length nvar, input x buffer
  vbuf::V     # length ncon, cotangent seed buffer
  jtvbuf::V   # length nvar, output Jtv buffer
end

function EnzymeReverseADJtprod(
  nvar::Integer,
  f,
  ncon::Integer = 0,
  c::Function = (args...) -> [];
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  cx = fill!(similar(x0, ncon), zero(T))
  xbuf = similar(x0)
  vbuf = fill!(similar(x0, ncon), zero(T))
  jtvbuf = similar(x0)
  return EnzymeReverseADJtprod(cx, xbuf, vbuf, jtvbuf)
end

struct SparseEnzymeADJacobian{R, C, S, V} <: ADBackend
  nvar::Int
  ncon::Int
  rowval::Vector{Int}
  colptr::Vector{Int}
  nzval::Vector{R}
  result_coloring::C
  compressed_jacobian::S
  v::V
  cx::V
  xbuf::V
end

function SparseEnzymeADJacobian(
  nvar,
  f,
  ncon,
  c!;
  x0::AbstractVector = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:direct}(
    postprocessing = true,
  ),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  show_time::Bool = false,
  kwargs...,
)
  timer = @elapsed begin
    output = similar(x0, ncon)
    J = compute_jacobian_sparsity(c!, output, x0, detector = detector)
  end
  show_time && println("  • Sparsity pattern detection of the Jacobian: $timer seconds.")
  SparseEnzymeADJacobian(nvar, f, ncon, c!, J; x0, coloring_algorithm, show_time, kwargs...)
end

function SparseEnzymeADJacobian(
  nvar,
  f,
  ncon,
  c!,
  J::SparseMatrixCSC{Bool, Int};
  x0::AbstractVector{T} = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:direct}(
    postprocessing = true,
  ),
  show_time::Bool = false,
  kwargs...,
) where {T}
  timer = @elapsed begin
    # We should support :row and :bidirectional in the future
    problem = ColoringProblem{:nonsymmetric, :column}()
    result_coloring = coloring(J, problem, coloring_algorithm, decompression_eltype = T)

    rowval = J.rowval
    colptr = J.colptr
    nzval = T.(J.nzval)
    compressed_jacobian = similar(x0, ncon)
  end
  show_time && println("  • Coloring of the sparse Jacobian: $timer seconds.")

  timer = @elapsed begin
    v = similar(x0)
    cx = fill!(similar(x0, ncon), zero(T))
    xbuf = similar(x0)
  end
  show_time && println("  • Allocation of the AD buffers for the sparse Jacobian: $timer seconds.")

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
    xbuf,
  )
end

struct SparseEnzymeADHessian{R, C, S, L, F, V} <: ADBackend
  nvar::Int
  rowval::Vector{Int}
  colptr::Vector{Int}
  nzval::Vector{R}
  result_coloring::C
  coloring_mode::Symbol
  compressed_hessian_icol::V
  compressed_hessian::S
  v::V
  y::V
  grad::V
  cx::V
  f::F
  ℓ::L
  xbuf::V
end

function SparseEnzymeADHessian(
  nvar,
  f,
  ncon,
  c!;
  x0::AbstractVector = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:substitution}(
    postprocessing = true,
  ),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  show_time::Bool = false,
  kwargs...,
)
  timer = @elapsed begin
    H = compute_hessian_sparsity(f, nvar, c!, ncon, detector = detector)
  end
  show_time && println("  • Sparsity pattern detection of the Hessian: $timer seconds.")
  SparseEnzymeADHessian(nvar, f, ncon, c!, H; x0, coloring_algorithm, show_time, kwargs...)
end

function SparseEnzymeADHessian(
  nvar,
  f,
  ncon,
  c!,
  H::SparseMatrixCSC{Bool, Int};
  x0::AbstractVector{T} = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:substitution}(
    postprocessing = true,
  ),
  show_time::Bool = false,
  kwargs...,
) where {T}
  timer = @elapsed begin
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
  end
  show_time && println("  • Coloring of the sparse Hessian: $timer seconds.")

  timer = @elapsed begin
    v = similar(x0)
    y = similar(x0, ncon)
    cx = similar(x0, ncon)
    grad = similar(x0)
    xbuf = similar(x0)

    function ℓ(x, y, obj_weight, cx)
      if ncon != 0
        c!(cx, x)
      end
      res = obj_weight * f(x)
      if ncon != 0
        for i = 1:ncon
          res += cx[i] * y[i]
        end
      end
      return res
    end
  end
  show_time && println("  • Allocation of the AD buffers for the sparse Hessian: $timer seconds.")

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
    f,
    ℓ,
    xbuf,
  )
end
