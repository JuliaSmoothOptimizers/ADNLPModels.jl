# Brown and Dennis functions
#
#   Source: Problem 16 in
#   J.J. Moré, B.S. Garbow and K.E. Hillstrom,
#   "Testing Unconstrained Optimization Software",
#   ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981
#
#   classification SUR2-AN-4-0
function brownden_radnlp(; n :: Int = 4, type :: Val{T} = Val(Float64)) where T

  x0 = convert(Array{T}, [25; 5; -5; -1])
  f(x) = begin
    s = zero(T)
    for i = 1:20
      s += ((x[1] + x[2] * T(i)/5 - exp(T(i)/5))^2 + (x[3] + x[4] * sin(T(i)/5) - cos(T(i)/5))^2)^2
    end
    return s
  end

  return RADNLPModel(f, x0, name="brownden_radnlp")
end

function brownden_autodiff(; n :: Int = 4, type :: Val{T} = Val(Float64)) where T

  x0 = convert(Array{T}, [25; 5; -5; -1])
  f(x) = begin
    s = zero(T)
    for i = 1:20
      s += ((x[1] + x[2] * T(i)/5 - exp(T(i)/5))^2 + (x[3] + x[4] * sin(T(i)/5) - cos(T(i)/5))^2)^2
    end
    return s
  end

  return ADNLPModel(f, x0, name="brownden_autodiff")
end

mutable struct BROWNDEN <: AbstractNLPModel
  meta :: NLPModelMeta
  counters :: Counters
end

function BROWNDEN()
  meta = NLPModelMeta(4, x0=[25.0; 5.0; -5.0; -1.0], name="BROWNDEN_manual", nnzh=10)

  return BROWNDEN(meta, Counters())
end

function ADNLPModels.obj(nlp :: BROWNDEN, x :: AbstractVector{T}) where T
  @lencheck 4 x
  increment!(nlp, :neval_obj)
  return sum(((x[1] + x[2] * T(i)/5 - exp(T(i)/5))^2 + (x[3] + x[4] * sin(T(i)/5) - cos(T(i)/5))^2)^2 for i = 1:20)
end

function ADNLPModels.grad!(nlp :: BROWNDEN, x :: AbstractVector, gx :: AbstractVector)
  @lencheck 4 x gx
  increment!(nlp, :neval_grad)
  α(x,i) = x[1] + x[2] * i/5 - exp(i/5)
  β(x,i) = x[3] + x[4] * sin(i/5) - cos(i/5)
  θ(x,i) = α(x,i)^2 + β(x,i)^2
  gx .= sum(4 * θ(x,i) * (α(x,i) * [1; i/5; 0; 0] + β(x,i) * [0; 0; 1; sin(i/5)]) for i = 1:20)
  return gx
end

function ADNLPModels.hess(nlp :: BROWNDEN, x :: AbstractVector{T}; obj_weight=1.0) where T
  @lencheck 4 x
  increment!(nlp, :neval_hess)
  α(x,i) = x[1] + x[2] * T(i)/5 - exp(T(i)/5)
  β(x,i) = x[3] + x[4] * sin(T(i)/5) - cos(T(i)/5)
  Hx = zeros(T, 4, 4)
  if obj_weight == 0
    return Hx
  end
  for i = 1:20
    αi, βi = α(x,i), β(x,i)
    vi, wi = T[1; i/5; 0; 0], T[0; 0; 1; sin(i/5)]
    zi = αi * vi + βi * wi
    θi = αi^2 + βi^2
    Hx += (4vi * vi' + 4wi * wi') * θi + 8zi * zi'
  end
  return T(obj_weight) * tril(Hx)
end

function ADNLPModels.hess_structure!(nlp :: BROWNDEN, rows :: AbstractVector{Int}, cols :: AbstractVector{Int})
  @lencheck 10 rows cols
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function ADNLPModels.hess_coord!(nlp :: BROWNDEN, x :: AbstractVector, vals :: AbstractVector; obj_weight=1.0)
  @lencheck 4 x
  @lencheck 10 vals
  Hx = hess(nlp, x, obj_weight=obj_weight)
  k = 1
  for j = 1:4
    for i = j:4
      vals[k] = Hx[i,j]
      k += 1
    end
  end
  return vals
end

function ADNLPModels.hprod!(nlp :: BROWNDEN, x :: AbstractVector{T}, v :: AbstractVector{T}, Hv :: AbstractVector{T}; obj_weight=one(T)) where T
  @lencheck 4 x v Hv
  increment!(nlp, :neval_hprod)
  α(x,i) = x[1] + x[2] * i/5 - exp(i/5)
  β(x,i) = x[3] + x[4] * sin(i/5) - cos(i/5)
  Hv .= 0
  if obj_weight == 0
    return Hv
  end
  for i = 1:20
    αi, βi = α(x,i), β(x,i)
    vi, wi = [1; i/5; 0; 0], [0; 0; 1; sin(i/5)]
    zi = αi * vi + βi * wi
    θi = αi^2 + βi^2
    Hv .+= obj_weight * ((4 * dot(vi, v) * vi + 4 * dot(wi, v) * wi) * θi + 8 * dot(zi, v) * zi)
  end
  return Hv
end