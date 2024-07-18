struct SparseADHessian{Tag, GT, S, T} <: ADNLPModels.ADBackend
  d::BitVector
  rowval::Vector{Int}
  colptr::Vector{Int}
  colors::Vector{Int}
  ncolors::Int
  dcolors::Dict{Int, Vector{Tuple{Int,Int}}}
  res::S
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
  x0::S = rand(nvar),
  coloring::AbstractColoringAlgorithm = GreedyColoringAlgorithm(),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  kwargs...,
) where {S}
  T = eltype(S)
  H = compute_hessian_sparsity(f, nvar, c!, ncon, detector = detector)

  colors, star_set = SparseMatrixColorings.symmetric_coloring_detailed(H, coloring)
  ncolors = maximum(colors)

  d = BitVector(undef, nvar)

  trilH = tril(H)
  rowval = trilH.rowval
  colptr = trilH.colptr

  # The indices of the nonzero elements in `vals` that will be processed by color `c` are stored in `dcolors[c]`.
  dcolors = nnz_colors(trilH, star_set, colors, ncolors)

  # prepare directional derivatives
  res = similar(x0)

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

  return SparseADHessian(
    d,
    rowval,
    colptr,
    colors,
    ncolors,
    dcolors,
    res,
    lz,
    glz,
    sol,
    longv,
    Hvp,
    ∇φ!,
    y,
  )
end

struct SparseReverseADHessian{T, S, Tagf, F, Tagψ, P} <: ADNLPModels.ADBackend
  d::BitVector
  rowval::Vector{Int}
  colptr::Vector{Int}
  colors::Vector{Int}
  ncolors::Int
  dcolors::Dict{Int, Vector{Tuple{Int,Int}}}
  res::S
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
  x0::AbstractVector{T} = rand(nvar),
  coloring::AbstractColoringAlgorithm = GreedyColoringAlgorithm(),
  detector::AbstractSparsityDetector = TracerSparsityDetector(),
  kwargs...,
) where {T}
  H = compute_hessian_sparsity(f, nvar, c!, ncon, detector = detector)

  colors, star_set = SparseMatrixColorings.symmetric_coloring_detailed(H, coloring)
  ncolors = maximum(colors)

  d = BitVector(undef, nvar)

  trilH = tril(H)
  rowval = trilH.rowval
  colptr = trilH.colptr

  # The indices of the nonzero elements in `vals` that will be processed by color `c` are stored in `dcolors[c]`.
  dcolors = nnz_colors(trilH, star_set, colors, ncolors)

  # prepare directional derivatives
  res = similar(x0)

  # unconstrained Hessian
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
  return SparseReverseADHessian(
    d,
    rowval,
    colptr,
    colors,
    ncolors,
    dcolors,
    res,
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
  b::SparseADHessian{Tag, GT, S, T},
  x::AbstractVector,
  obj_weight,
  y::AbstractVector,
  vals::AbstractVector,
) where {Tag, GT, S, T}
  nvar = length(x)
  ncon = length(y)

  b.sol[1:ncon] .= zero(T) # cx
  b.sol[(ncon + 1):(ncon + nvar)] .= x
  b.sol[(ncon + nvar + 1):(2 * ncon + nvar)] .= y
  b.sol[end] = obj_weight

  b.longv .= 0

  for icol = 1:(b.ncolors)
    dcolor_icol = b.dcolors[icol]
    if !isempty(dcolor_icol)
      b.d .= (b.colors .== icol)
      b.longv[(ncon + 1):(ncon + nvar)] .= b.d
      map!(ForwardDiff.Dual{Tag}, b.lz, b.sol, b.longv)
      b.∇φ!(b.glz, b.lz)
      ForwardDiff.extract_derivative!(Tag, b.Hvp, b.glz)
      b.res .= view(b.Hvp, (ncon + 1):(ncon + nvar))

      # Store in `vals` the nonzeros of each column of the Hessian computed with color `icol`
      for (row, k) in dcolor_icol
        vals[k] = b.res[row]
      end
    end
  end

  return vals
end

function sparse_hess_coord!(
  b::SparseReverseADHessian{T, S, Tagf, F, Tagψ, P},
  x::AbstractVector,
  obj_weight,
  y::AbstractVector,
  vals::AbstractVector,
) where {T, S, Tagf, F, Tagψ, P}
  nvar = length(x)

  for icol = 1:(b.ncolors)
    dcolor_icol = b.dcolors[icol]
    if !isempty(dcolor_icol)
      b.d .= (b.colors .== icol)

      # objective
      map!(ForwardDiff.Dual{Tagf}, b.z, x, b.d) # x + ε * v
      b.∇f!(b.gz, b.z)
      ForwardDiff.extract_derivative!(Tagf, b.res, b.gz)
      b.res .*= obj_weight

      # constraints
      map!(ForwardDiff.Dual{Tagψ}, b.zψ, x, b.d)
      b.yψ .= y
      b.∇l!(b.gzψ, b.gyψ, b.zψ, b.yψ)
      ForwardDiff.extract_derivative!(Tagψ, b.Hv_temp, b.gzψ)
      b.res .+= b.Hv_temp

      # Store in `vals` the nonzeros of each column of the Hessian computed with color `icol`
      for (row, k) in dcolor_icol
        vals[k] = b.res[row]
      end
    end
  end

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
