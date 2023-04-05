## ----- SparseDiffTools -----

struct SparseForwardADHessian{T, T2, T3, T4} <: ADNLPModels.ADBackend
  cfH::ForwardAutoColorHesCache{T, T2, T3, T4}
end

function SparseForwardADHessian(nvar, f, ncon, c!;
  x0=rand(nvar),
  alg::SparseDiffTools.SparseDiffToolsColoringAlgorithm = SparseDiffTools.GreedyD1Color(),
  kwargs...,
)
  @variables xs[1:nvar], μs
  xsi = Symbolics.scalarize(xs)
  fun = μs * f(xsi)
  if ncon > 0
    @variables ys[1:ncon]
    ysi = Symbolics.scalarize(ys)
    cx = similar(ysi)
    fun = fun + dot(c!(cx,xsi), ysi)
  end
  S = Symbolics.hessian_sparsity(fun, ncon == 0 ? xsi : [xsi; ysi], full=false)
  H = ncon == 0 ? S : S[1:nvar,1:nvar]
  rows, cols, _ = findnz(H)
  colors = matrix_colors(H, alg)
  Tv = eltype(x0)
  hess = SparseMatrixCSC{Tv, Int}(H.m, H.n, H.colptr, H.rowval, Tv.(H.nzval))

  # Create a ForwardColorJacCache
  chunksize = ForwardDiff.pickchunksize(maximum(colors))
  chunk = ForwardDiff.Chunk(chunksize)
  tag = ForwardDiff.Tag(SparseDiffTools.AutoAutoTag(), Tv)
  jacobian_config = ForwardDiff.JacobianConfig(f, x0, chunk, tag)
  gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals, chunk, tag)
  outer_tag = SparseDiffTools.get_tag(jacobian_config.duals)
  g! = (G, x) -> ForwardDiff.gradient!(G, f, x, gradient_config, Val(false))
  jac_cache = ForwardColorJacCache(g!, x0; colorvec=colors, sparsity=hess, tag=outer_tag)

  cfH = ForwardAutoColorHesCache(jac_cache, g!, hess, colors)
  return SparseForwardADHessian(cfH)
end

function get_nln_nnzh(b::SparseForwardADHessian, nvar)
  nnz(b.cfH.sparsity)
end

function hess_structure!(
  b::SparseForwardADHessian,
  nlp::ADModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= rowvals(b.cfH.sparsity)
  for i = 1:(nlp.meta.nvar)
    for j = b.cfH.sparsity.colptr[i]:(b.cfH.sparsity.colptr[i + 1] - 1)
      cols[j] = i
    end
  end
  return rows, cols
end

function hess_coord!(
  b::SparseForwardADHessian,
  nlp::ADModel,
  x::AbstractVector,
  y::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ = get_lag(nlp, b, obj_weight, y)
  autoauto_color_hessian!(b.cfH.sparsity, ℓ, x, b.cfH)
  vals .= nonzeros(b.cfH.sparsity)
  return vals
end

function hess_coord!(
  b::SparseForwardADHessian,
  nlp::ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ(x) = obj_weight * nlp.f(x)
  autoauto_color_hessian!(b.cfH.sparsity, ℓ, x, b.cfH)
  vals .= nonzeros(b.cfH.sparsity)
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

function SparseSymbolicsADHessian(nvar, f, ncon, c!; x0::AbstractVector{T} = rand(nvar), kwargs...) where {T}
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
