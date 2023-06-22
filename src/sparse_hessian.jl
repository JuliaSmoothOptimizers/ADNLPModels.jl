struct SparseADHessian{Tag, GT, S, T} <: ADNLPModels.ADBackend
  d::BitVector
  rowval::Vector{Int}
  colptr::Vector{Int}
  colors::Vector{Int}
  ncolors::Int
  res::S
  lz::Vector{ForwardDiff.Dual{Tag, T, 1}}
  glz::Vector{ForwardDiff.Dual{Tag, T, 1}}
  sol::S
  longv::S
  Hvp::S
  ∇φ!::GT
end

function SparseADHessian(nvar, f, ncon, c!; x0::AbstractVector{T} = rand(nvar), alg = ColPackColoration(), kwargs...) where {T}
  @variables xs[1:nvar]
  xsi = Symbolics.scalarize(xs)
  fun = f(xsi)
  if ncon > 0
    @variables ys[1:ncon]
    ysi = Symbolics.scalarize(ys)
    cx = similar(ysi)
    fun = fun + dot(c!(cx, xsi), ysi)
  end
  S = Symbolics.hessian_sparsity(fun, ncon == 0 ? xsi : [xsi; ysi]) # , full = false
  H = ncon == 0 ? S : S[1:nvar, 1:nvar]

  colors = sparse_matrix_colors(H, alg)
  ncolors = maximum(colors)

  d = BitVector(undef, nvar)

  trilH = tril(H)
  rowval = trilH.rowval
  colptr = trilH.colptr

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
  longv = zeros(T, ntotal)
  Hvp = zeros(T, ntotal)

  return SparseADHessian(d, rowval, colptr, colors, ncolors, res, lz, glz, sol, longv, Hvp, ∇φ!)
end

function get_nln_nnzh(b::SparseADHessian, nvar)
  return length(b.rowval)
end

function hess_structure!(
  b::SparseADHessian,
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
  ℓ::Function,
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
    b.d .= (b.colors .== icol)
    b.longv[(ncon + 1):(ncon + nvar)] .= b.d
    map!(ForwardDiff.Dual{Tag}, b.lz, b.sol, b.longv)
    b.∇φ!(b.glz, b.lz)
    ForwardDiff.extract_derivative!(Tag, b.Hvp, b.glz)
    b.res .= view(b.Hvp, (ncon + 1):(ncon + nvar))
    for j = 1:nvar
      if b.colors[j] == icol
        for k = b.colptr[j]:(b.colptr[j + 1] - 1)
          i = b.rowval[k]
          vals[k] = b.res[i]
        end
      end
    end
  end

  return vals
end

function hess_coord!(
  b::SparseADHessian,
  nlp::ADModel,
  x::AbstractVector,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ = get_lag(nlp, b, obj_weight, y)
  sparse_hess_coord!(ℓ, b, x, obj_weight, y, vals)
end

function hess_coord!(
  b::SparseADHessian,
  nlp::ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  y = zeros(eltype(x), nlp.meta.nnln)
  ℓ = get_lag(nlp, b, obj_weight, y)
  sparse_hess_coord!(ℓ, b, x, obj_weight, y, vals)
end

function hess_coord!(
  b::SparseADHessian,
  nlp::ADModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  y = zeros(T, nlp.meta.nnln)
  for (w, k) in enumerate(nlp.meta.nln)
    y[w] = k == j ? 1 : 0
  end
  obj_weight = zero(T)
  ℓ = get_lag(nlp, b, obj_weight, y)
  sparse_hess_coord!(ℓ, b, x, obj_weight, y, vals)
  return vals
end

## ----- Symbolics -----

struct SparseSymbolicsADHessian{T, H} <: ADBackend
  nnzh::Int
  rows::Vector{Int}
  cols::Vector{Int}
  y::AbstractVector{T}
  cfH::H
end

function SparseSymbolicsADHessian(
  nvar,
  f,
  ncon,
  c!;
  x0::AbstractVector{T} = rand(nvar),
  kwargs...,
) where {T}
  @variables xs[1:nvar], μs
  xsi = Symbolics.scalarize(xs)
  fun = μs * f(xsi)
  @variables ys[1:ncon]
  ysi = Symbolics.scalarize(ys)
  if ncon > 0
    cx = similar(ysi)
    fun = fun + dot(c!(cx, xsi), ysi)
  end
  H = Symbolics.hessian_sparsity(fun, ncon == 0 ? xsi : [xsi; ysi], full = false)
  H = ncon == 0 ? H : H[1:nvar, 1:nvar]
  rows, cols, _ = findnz(H)
  vals = Symbolics.sparsehessian_vals(fun, xsi, rows, cols)
  nnzh = length(vals)
  # cfH is a Tuple{Expr, Expr}, cfH[2] is the in-place function
  # that we need to update a vector `vals` with the nonzeros of ∇²ℓ(x, y, μ).
  cfH = Symbolics.build_function(vals, xsi, ysi, μs, expression = Val{false})
  y = zeros(T, ncon)
  return SparseSymbolicsADHessian(nnzh, rows, cols, y, cfH[2])
end

function get_nln_nnzh(b::SparseSymbolicsADHessian, nvar)
  b.nnzh
end

function hess_structure!(
  b::SparseSymbolicsADHessian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= b.rows
  cols .= b.cols
  return rows, cols
end

function hess_coord!(
  b::SparseSymbolicsADHessian,
  nlp::ADModel,
  x::AbstractVector,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  @eval $(b.cfH)($vals, $x, $y, $obj_weight)
  return vals
end

function hess_coord!(
  b::SparseSymbolicsADHessian,
  nlp::ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  b.y .= 0
  @eval $(b.cfH)($vals, $x, $(b.y), $obj_weight)
  return vals
end

function hess_coord!(
  b::SparseSymbolicsADHessian,
  nlp::ADModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector{T},
) where {T}
  for (w, k) in enumerate(nlp.meta.nln)
    b.y[w] = k == j ? 1 : 0
  end
  obj_weight = zero(T)
  @eval $(b.cfH)($vals, $x, $(b.y), $obj_weight)
  return vals
end
