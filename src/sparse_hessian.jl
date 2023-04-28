struct SparseADHessian <: ADNLPModels.ADBackend
  d::BitVector
  rowval::Vector{Int}
  colptr::Vector{Int}
  colors::Vector{Int}
  ncolors::Int
end

function SparseADHessian(nvar, f, ncon, c!;
  x0=rand(nvar),
  alg::SparseDiffTools.SparseDiffToolsColoringAlgorithm = SparseDiffTools.GreedyD1Color(),
  kwargs...,
)
  @variables xs[1:nvar]
  xsi = Symbolics.scalarize(xs)
  fun = f(xsi)
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
  d = BitVector(undef, nvar)
  ncolors = maximum(colors)
  return SparseADHessian(d, H.rowval, H.colptr, colors, ncolors)
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
  b::SparseADHessian,
  x::AbstractVector,
  vals::AbstractVector
  )
  nvar = length(x)
  for icol = 1 : b.ncolors
    b.d .= (b.colors .== icol)
    res = ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * b.d), 0)
    for j = 1 : nvar
      if b.colors[j] == icol
        for k = b.colptr[j] : b.colptr[j+1] - 1
          i = b.rowval[k]
          vals[k] = res[i]
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
  sparse_hess_coord!(ℓ, b, x, vals)
end

function hess_coord!(
  b::SparseADHessian,
  nlp::ADModel,
  x::AbstractVector,
  obj_weight::Real,
  vals::AbstractVector,
)
  ℓ(x) = obj_weight * nlp.f(x)
  sparse_hess_coord!(ℓ, b, x, vals)
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
