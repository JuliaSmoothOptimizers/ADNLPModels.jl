struct SparseADHessian{Tag, R, T, C, H, S, GT} <: ADBackend
  nvar::Int
  rowval::Vector{Int}
  colptr::Vector{Int}
  nzval::Vector{R}
  result_coloring::C
  coloring_mode::Symbol
  compressed_hessian::H
  seed::BitVector
  lz::Vector{ForwardDiff.Dual{Tag, T, 1}}
  glz::Vector{ForwardDiff.Dual{Tag, T, 1}}
  sol::S
  longv::S
  Hvp::S
  ∇φ!::GT
  y::S
end

function SparseADHessian(
  nvar,
  f,
  ncon,
  c!;
  x0::AbstractVector = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:direct}(),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  show_time::Bool = false,
  kwargs...,
)
  timer = @elapsed begin
    H = compute_hessian_sparsity(f, nvar, c!, ncon, detector = detector)
  end
  show_time && println("  • Sparsity pattern detection of the Hessian: $timer seconds.")
  SparseADHessian(nvar, f, ncon, c!, H; x0, coloring_algorithm, show_time, kwargs...)
end

function SparseADHessian(
  nvar,
  f,
  ncon,
  c!,
  H::SparseMatrixCSC{Bool, Int64};
  x0::S = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:direct}(),
  show_time::Bool = false,
  kwargs...,
) where {S}
  T = eltype(S)

  timer = @elapsed begin
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
    seed = BitVector(undef, nvar)
  end
  show_time && println("  • Coloring of the sparse Hessian: $timer seconds.")

  timer = @elapsed begin
    function lag(z; nvar = nvar, ncon = ncon, f = f, c! = c!)
      cx, x, y, ob = view(z, 1:ncon),
      view(z, (ncon + 1):(nvar + ncon)),
      view(z, (nvar + ncon + 1):(nvar + ncon + ncon)),
      z[end]
      if ncon > 0
        c!(cx, x)
        return ob * f(x) + dot(cx, y)
      else
        return ob * f(x)
      end
    end

    ntotal = nvar + 2 * ncon + 1
    sol = similar(x0, ntotal)
    lz = Vector{ForwardDiff.Dual{ForwardDiff.Tag{typeof(lag), T}, T, 1}}(undef, ntotal)
    glz = similar(lz)
    cfg = ForwardDiff.GradientConfig(lag, lz)
    function ∇φ!(gz, z; lag = lag, cfg = cfg)
      ForwardDiff.gradient!(gz, lag, z, cfg)
      return gz
    end
    longv = fill!(S(undef, ntotal), 0)
    Hvp = fill!(S(undef, ntotal), 0)
    y = fill!(S(undef, ncon), 0)
  end
  show_time && println("  • Allocation of the AD buffers for the sparse Hessian: $timer seconds.")

  return SparseADHessian(
    nvar,
    rowval,
    colptr,
    nzval,
    result_coloring,
    coloring_mode,
    compressed_hessian,
    seed,
    lz,
    glz,
    sol,
    longv,
    Hvp,
    ∇φ!,
    y,
  )
end

struct SparseReverseADHessian{Tagf, Tagψ, R, T, C, H, S, F, P} <: ADBackend
  nvar::Int
  rowval::Vector{Int}
  colptr::Vector{Int}
  nzval::Vector{R}
  result_coloring::C
  coloring_mode::Symbol
  compressed_hessian::H
  seed::BitVector
  z::Vector{ForwardDiff.Dual{Tagf, T, 1}}
  gz::Vector{ForwardDiff.Dual{Tagf, T, 1}}
  ∇f!::F
  zψ::Vector{ForwardDiff.Dual{Tagψ, T, 1}}
  yψ::Vector{ForwardDiff.Dual{Tagψ, T, 1}}
  gzψ::Vector{ForwardDiff.Dual{Tagψ, T, 1}}
  gyψ::Vector{ForwardDiff.Dual{Tagψ, T, 1}}
  ∇l!::P
  Hv_temp::S
  y::S
end

function SparseReverseADHessian(
  nvar,
  f,
  ncon,
  c!;
  x0::AbstractVector = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:substitution}(),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  show_time::Bool = false,
  kwargs...,
)
  timer = @elapsed begin
    H = compute_hessian_sparsity(f, nvar, c!, ncon, detector = detector)
  end
  show_time && println("  • Sparsity pattern detection of the Hessian: $timer seconds.")
  SparseReverseADHessian(nvar, f, ncon, c!, H; x0, coloring_algorithm, show_time, kwargs...)
end

function SparseReverseADHessian(
  nvar,
  f,
  ncon,
  c!,
  H::SparseMatrixCSC{Bool, Int};
  x0::AbstractVector{T} = rand(nvar),
  coloring_algorithm::AbstractColoringAlgorithm = GreedyColoringAlgorithm{:substitution}(),
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
      compressed_hessian = similar(x0)
    else
      coloring_mode = :substitution
      group = column_groups(result_coloring)
      ncolors = length(group)
      compressed_hessian = similar(x0, (nvar, ncolors))
    end
    seed = BitVector(undef, nvar)
  end
  show_time && println("  • Coloring of the sparse Hessian: $timer seconds.")

  # unconstrained Hessian
  timer = @elapsed begin
    tagf = ForwardDiff.Tag{typeof(f), T}
    z = Vector{ForwardDiff.Dual{tagf, T, 1}}(undef, nvar)
    gz = similar(z)
    f_tape = ReverseDiff.GradientTape(f, z)
    cfgf = ReverseDiff.compile(f_tape)
    ∇f!(gz, z; cfg = cfgf) = ReverseDiff.gradient!(gz, cfg, z)

    # constraints
    ψ(x, u) = begin # ; tmp_out = _tmp_out
      ncon = length(u)
      tmp_out = similar(x, ncon)
      c!(tmp_out, x)
      dot(tmp_out, u)
    end
    tagψ = ForwardDiff.Tag{typeof(ψ), T}
    zψ = Vector{ForwardDiff.Dual{tagψ, T, 1}}(undef, nvar)
    yψ = fill!(similar(zψ, ncon), zero(T))
    ψ_tape = ReverseDiff.GradientConfig((zψ, yψ))
    cfgψ = ReverseDiff.compile(ReverseDiff.GradientTape(ψ, (zψ, yψ), ψ_tape))

    gzψ = similar(zψ)
    gyψ = similar(yψ)
    function ∇l!(gz, gy, z, y; cfg = cfgψ)
      ReverseDiff.gradient!((gz, gy), cfg, (z, y))
    end
    Hv_temp = similar(x0)
    y = similar(x0, ncon)
  end
  show_time && println("  • Allocation of the AD buffers for the sparse Hessian: $timer seconds.")

  return SparseReverseADHessian(
    nvar,
    rowval,
    colptr,
    nzval,
    result_coloring,
    coloring_mode,
    compressed_hessian,
    seed,
    z,
    gz,
    ∇f!,
    zψ,
    yψ,
    gzψ,
    gyψ,
    ∇l!,
    Hv_temp,
    y,
  )
end

function get_nln_nnzh(b::Union{SparseADHessian, SparseReverseADHessian}, nvar)
  return length(b.rowval)
end

function NLPModels.hess_structure!(
  b::Union{SparseADHessian, SparseReverseADHessian},
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

function NLPModels.hess_structure_residual!(
  b::Union{SparseADHessian, SparseReverseADHessian},
  nls::AbstractADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  return hess_structure!(b, nls, rows, cols)
end

function sparse_hess_coord!(
  b::SparseADHessian{Tag},
  x::AbstractVector,
  obj_weight,
  y::AbstractVector,
  vals::AbstractVector,
) where {Tag}
  ncon = length(y)
  T = eltype(x)
  b.sol[1:ncon] .= zero(T)  # cx
  b.sol[(ncon + 1):(ncon + b.nvar)] .= x
  b.sol[(ncon + b.nvar + 1):(2 * ncon + b.nvar)] .= y
  b.sol[end] = obj_weight

  b.longv .= 0

  # SparseMatrixColorings.jl requires a SparseMatrixCSC for the decompression
  A = SparseMatrixCSC(b.nvar, b.nvar, b.colptr, b.rowval, b.nzval)

  groups = column_groups(b.result_coloring)
  for (icol, cols) in enumerate(groups)
    # Update the seed
    b.seed .= false
    for col in cols
      b.seed[col] = true
    end

    # column icol of the compressed hessian
    compressed_hessian_icol =
      (b.coloring_mode == :direct) ? b.compressed_hessian : view(b.compressed_hessian, :, icol)

    b.longv[(ncon + 1):(ncon + b.nvar)] .= b.seed
    map!(ForwardDiff.Dual{Tag}, b.lz, b.sol, b.longv)
    b.∇φ!(b.glz, b.lz)
    ForwardDiff.extract_derivative!(Tag, b.Hvp, b.glz)
    compressed_hessian_icol .= view(b.Hvp, (ncon + 1):(ncon + b.nvar))
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

function sparse_hess_coord!(
  b::SparseReverseADHessian{Tagf, Tagψ},
  x::AbstractVector,
  obj_weight,
  y::AbstractVector,
  vals::AbstractVector,
) where {Tagf, Tagψ}
  # SparseMatrixColorings.jl requires a SparseMatrixCSC for the decompression
  A = SparseMatrixCSC(b.nvar, b.nvar, b.colptr, b.rowval, b.nzval)

  groups = column_groups(b.result_coloring)
  for (icol, cols) in enumerate(groups)
    # Update the seed
    b.seed .= false
    for col in cols
      b.seed[col] = true
    end

    # column icol of the compressed hessian
    compressed_hessian_icol =
      (b.coloring_mode == :direct) ? b.compressed_hessian : view(b.compressed_hessian, :, icol)

    # objective
    map!(ForwardDiff.Dual{Tagf}, b.z, x, b.seed)  # x + ε * v
    b.∇f!(b.gz, b.z)
    ForwardDiff.extract_derivative!(Tagf, compressed_hessian_icol, b.gz)
    compressed_hessian_icol .*= obj_weight

    # constraints
    map!(ForwardDiff.Dual{Tagψ}, b.zψ, x, b.seed)
    b.yψ .= y
    b.∇l!(b.gzψ, b.gyψ, b.zψ, b.yψ)
    ForwardDiff.extract_derivative!(Tagψ, b.Hv_temp, b.gzψ)
    compressed_hessian_icol .+= b.Hv_temp

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
  b::Union{SparseADHessian, SparseReverseADHessian},
  nlp::ADModel,
  x::AbstractVector,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  sparse_hess_coord!(b, x, obj_weight, y, vals)
end

function NLPModels.hess_coord!(
  b::Union{SparseADHessian, SparseReverseADHessian},
  nlp::ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  b.y .= 0
  sparse_hess_coord!(b, x, obj_weight, b.y, vals)
end

function NLPModels.hess_coord!(
  b::Union{SparseADHessian, SparseReverseADHessian},
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

function NLPModels.hess_coord_residual!(
  b::Union{SparseADHessian, SparseReverseADHessian},
  nls::AbstractADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  obj_weight = zero(eltype(x))
  sparse_hess_coord!(b, x, obj_weight, v, vals)
end
