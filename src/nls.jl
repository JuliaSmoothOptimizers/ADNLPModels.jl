export ADNLSModel, ADNLSModel!

mutable struct ADNLSModel{T, S, Si, F1, F2, ADMB <: ADModelBackend} <: AbstractADNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
  adbackend::ADMB

  # Function
  F!::F1

  clinrows::Si
  clincols::Si
  clinvals::S

  c!::F2
end

ADNLSModel(
  meta::NLPModelMeta{T, S},
  nls_meta::NLSMeta{T, S},
  counters::NLSCounters,
  adbackend::ADModelBackend,
  F,
  c,
) where {T, S} = ADNLSModel(meta, nls_meta, counters, adbackend, F, Int[], Int[], S(undef, 0), c)

ADNLPModels.show_header(io::IO, nls::ADNLSModel) = println(
  io,
  "ADNLSModel - Nonlinear least-squares model with automatic differentiation backend $(nls.adbackend)",
)

"""
    ADNLSModel(F, x0, nequ)
    ADNLSModel(F, x0, nequ, lvar, uvar)
    ADNLSModel(F, x0, nequ, clinrows, clincols, clinvals, lcon, ucon)
    ADNLSModel(F, x0, nequ, A, lcon, ucon)
    ADNLSModel(F, x0, nequ, c, lcon, ucon)
    ADNLSModel(F, x0, nequ, clinrows, clincols, clinvals, c, lcon, ucon)
    ADNLSModel(F, x0, nequ, A, c, lcon, ucon)
    ADNLSModel(F, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, lcon, ucon)
    ADNLSModel(F, x0, nequ, lvar, uvar, A, lcon, ucon)
    ADNLSModel(F, x0, nequ, lvar, uvar, c, lcon, ucon)
    ADNLSModel(F, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, c, lcon, ucon)
    ADNLSModel(F, x0, nequ, lvar, uvar, A, c, lcon, ucon)
    ADNLSModel(model::AbstractNLSModel)

ADNLSModel is an Nonlinear Least Squares model using automatic differentiation to
compute the derivatives.
The problem is defined as

     min  ½‖F(x)‖²
    s.to  lcon ≤ (  Ax  ) ≤ ucon
                 ( c(x) )
          lvar ≤   x  ≤ uvar

where `nequ` is the size of the vector `F(x)` and the linear constraints come first.

The following keyword arguments are available to all constructors:

- `linequ`: An array of indexes of the linear equations (default: `Int[]`)
- `minimize`: A boolean indicating whether this is a minimization problem (default: true)
- `name`: The name of the model (default: "Generic")

The following keyword arguments are available to the constructors for constrained problems:

- `y0`: An inital estimate to the Lagrangian multipliers (default: zeros)

`ADNLSModel` uses `ForwardDiff` and `ReverseDiff` for the automatic differentiation.
One can specify a new backend with the keyword arguments `backend::ADNLPModels.ADBackend`.
There are three pre-coded backends:
- the default `ForwardDiffAD`.
- `ReverseDiffAD`.
- `ZygoteDiffAD` accessible after loading `Zygote.jl` in your environment.
For an advanced usage, one can define its own backend and redefine the API as done in [ADNLPModels.jl/src/forward.jl](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl/blob/main/src/forward.jl).

# Examples
```julia
using ADNLPModels
F(x) = [x[2]; x[1]]
nequ = 2
x0 = ones(3)
nvar = 3
ADNLSModel(F, x0, nequ) # uses the default ForwardDiffAD backend.
ADNLSModel(F, x0, nequ; backend = ADNLPModels.ReverseDiffAD) # uses ReverseDiffAD backend.

using Zygote
ADNLSModel(F, x0, nequ; backend = ADNLPModels.ZygoteAD)
```

```julia
using ADNLPModels
F(x) = [x[2]; x[1]]
nequ = 2
x0 = ones(3)
c(x) = [1x[1] + x[2]; x[2]]
nvar, ncon = 3, 2
ADNLSModel(F, x0, nequ, c, zeros(ncon), zeros(ncon)) # uses the default ForwardDiffAD backend.
ADNLSModel(F, x0, nequ, c, zeros(ncon), zeros(ncon); backend = ADNLPModels.ReverseDiffAD) # uses ReverseDiffAD backend.

using Zygote
ADNLSModel(F, x0, nequ, c, zeros(ncon), zeros(ncon); backend = ADNLPModels.ZygoteAD)
```

For in-place constraints and residual function, use one of the following constructors:

    ADNLSModel!(F!, x0, nequ)
    ADNLSModel!(F!, x0, nequ, lvar, uvar)
    ADNLSModel!(F!, x0, nequ, c!, lcon, ucon)
    ADNLSModel!(F!, x0, nequ, clinrows, clincols, clinvals, c!, lcon, ucon)
    ADNLSModel!(F!, x0, nequ, clinrows, clincols, clinvals, lcon, ucon)
    ADNLSModel!(F!, x0, nequ, A, c!, lcon, ucon)
    ADNLSModel!(F!, x0, nequ, A, lcon, ucon)
    ADNLSModel!(F!, x0, nequ, lvar, uvar, c!, lcon, ucon)
    ADNLSModel!(F!, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, c!, lcon, ucon)
    ADNLSModel!(F!, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, lcon, ucon)
    ADNLSModel!(F!, x0, nequ, lvar, uvar, A, c!, lcon, ucon)
    ADNLSModel!(F!, x0, nequ, lvar, uvar, A, clcon, ucon)
    ADNLSModel!(model::AbstractNLSModel)

where the constraint function has the signature `c!(output, input)`.

```julia
using ADNLPModels
function F!(output, x)
  output[1] = x[2]
  output[2] = x[1]
end
nequ = 2
x0 = ones(3)
function c!(output, x) 
  output[1] = 1x[1] + x[2]
  output[2] = x[2]
end
nvar, ncon = 3, 2
nls = ADNLSModel!(F!, x0, nequ, c!, zeros(ncon), zeros(ncon))
```
"""
function ADNLSModel(F, x0::S, nequ::Integer; kwargs...) where {S}
  function F!(output, x)
    Fx = F(x)
    for i = 1:nequ
      output[i] = Fx[i]
    end
    return output
  end

  return ADNLSModel!(F!, x0, nequ; kwargs...)
end

function ADNLSModel!(
  F!,
  x0::S,
  nequ::Integer;
  linequ::AbstractVector{<:Integer} = Int[],
  name::String = "Generic",
  minimize::Bool = true,
  kwargs...,
) where {S}
  T = eltype(S)
  nvar = length(x0)

  adbackend = ADModelNLSBackend(nvar, F!, nequ; x0 = x0, kwargs...)
  nnzh = get_nln_nnzh(adbackend, nvar)

  meta = NLPModelMeta{T, S}(nvar, x0 = x0, nnzh = nnzh, name = name, minimize = minimize)
  nls_nnzj = get_residual_nnzj(adbackend, nvar, nequ)
  nls_nnzh = get_residual_nnzh(adbackend, nvar)
  nls_meta = NLSMeta{T, S}(nequ, nvar, nnzj = nls_nnzj, nnzh = nls_nnzh, lin = linequ)
  return ADNLSModel(meta, nls_meta, NLSCounters(), adbackend, F!, (cx, x) -> cx)
end

function ADNLSModel(F, x0::S, nequ::Integer, lvar::S, uvar::S; kwargs...) where {S}
  function F!(output, x)
    Fx = F(x)
    for i = 1:nequ
      output[i] = Fx[i]
    end
    return output
  end

  return ADNLSModel!(F!, x0, nequ, lvar, uvar; kwargs...)
end

function ADNLSModel!(
  F!,
  x0::S,
  nequ::Integer,
  lvar::S,
  uvar::S;
  linequ::AbstractVector{<:Integer} = Int[],
  name::String = "Generic",
  minimize::Bool = true,
  kwargs...,
) where {S}
  T = eltype(S)
  nvar = length(x0)
  @lencheck nvar lvar uvar

  adbackend = ADModelNLSBackend(nvar, F!, nequ; x0 = x0, kwargs...)
  nnzh = get_nln_nnzh(adbackend, nvar)

  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    nnzh = nnzh,
    name = name,
    minimize = minimize,
  )
  nls_nnzj = get_residual_nnzj(adbackend, nvar, nequ)
  nls_nnzh = get_residual_nnzh(adbackend, nvar)
  nls_meta = NLSMeta{T, S}(nequ, nvar, nnzj = nls_nnzj, nnzh = nls_nnzh, lin = linequ)
  return ADNLSModel(meta, nls_meta, NLSCounters(), adbackend, F!, (cx, x) -> cx)
end

function ADNLSModel(F, x0::S, nequ::Integer, c, lcon::S, ucon::S; kwargs...) where {S}
  function F!(output, x)
    Fx = F(x)
    for i = 1:nequ
      output[i] = Fx[i]
    end
    return output
  end

  function c!(output, x)
    cx = c(x)
    for i = 1:length(cx)
      output[i] = cx[i]
    end
    return output
  end

  return ADNLSModel!(F!, x0, nequ, c!, lcon, ucon; kwargs...)
end

function ADNLSModel!(
  F!,
  x0::S,
  nequ::Integer,
  c!,
  lcon::S,
  ucon::S;
  y0::S = fill!(similar(lcon), zero(eltype(S))),
  linequ::AbstractVector{<:Integer} = Int[],
  name::String = "Generic",
  minimize::Bool = true,
  kwargs...,
) where {S}
  T = eltype(S)
  nvar = length(x0)
  ncon = length(lcon)
  @lencheck ncon ucon y0

  adbackend = ADModelNLSBackend(nvar, F!, nequ, ncon, c!; x0 = x0, kwargs...)

  nnzh = get_nln_nnzh(adbackend, nvar)
  nnzj = get_nln_nnzj(adbackend, nvar, ncon)

  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    nnzh = nnzh,
    nln_nnzj = nnzj,
    name = name,
    minimize = minimize,
  )
  nls_nnzj = get_residual_nnzj(adbackend, nvar, nequ)
  nls_nnzh = get_residual_nnzh(adbackend, nvar)
  nls_meta = NLSMeta{T, S}(nequ, nvar, nnzj = nls_nnzj, nnzh = nls_nnzh, lin = linequ)
  return ADNLSModel(meta, nls_meta, NLSCounters(), adbackend, F!, c!)
end

function ADNLSModel(
  F,
  x0::S,
  nequ::Integer,
  clinrows::Si,
  clincols::Si,
  clinvals::S,
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Si}
  function F!(output, x)
    Fx = F(x)
    for i = 1:nequ
      output[i] = Fx[i]
    end
    return output
  end
  return ADNLSModel!(F!, x0, nequ, clinrows, clincols, clinvals, lcon, ucon; kwargs...)
end

function ADNLSModel(
  F,
  x0::S,
  nequ::Integer,
  A::AbstractSparseMatrix{Tv, Ti},
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Tv, Ti}
  function F!(output, x)
    Fx = F(x)
    for i = 1:nequ
      output[i] = Fx[i]
    end
    return output
  end
  return ADNLSModel!(F!, x0, nequ, A, lcon, ucon; kwargs...)
end

function ADNLSModel(
  F,
  x0::S,
  nequ::Integer,
  clinrows::Si,
  clincols::Si,
  clinvals::S,
  c,
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Si}
  function F!(output, x)
    Fx = F(x)
    for i = 1:nequ
      output[i] = Fx[i]
    end
    return output
  end

  function c!(output, x)
    cx = c(x)
    for i = 1:length(cx)
      output[i] = cx[i]
    end
    return output
  end

  return ADNLSModel!(F!, x0, nequ, clinrows, clincols, clinvals, c!, lcon, ucon; kwargs...)
end

function ADNLSModel!(
  F!,
  x0::S,
  nequ::Integer,
  clinrows::Si,
  clincols::Si,
  clinvals::S,
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Si}
  return ADNLSModel!(
    F!,
    x0,
    nequ,
    clinrows,
    clincols,
    clinvals,
    (cx, x) -> cx,
    lcon,
    ucon;
    kwargs...,
  )
end

function ADNLSModel!(
  F!,
  x0::S,
  nequ::Integer,
  clinrows::Si,
  clincols::Si,
  clinvals::S,
  c!,
  lcon::S,
  ucon::S;
  y0::S = fill!(similar(lcon), zero(eltype(S))),
  linequ::AbstractVector{<:Integer} = Int[],
  name::String = "Generic",
  minimize::Bool = true,
  kwargs...,
) where {S, Si}
  T = eltype(S)
  nvar = length(x0)
  ncon = length(lcon)
  @lencheck ncon ucon y0

  nlin = isempty(clinrows) ? 0 : maximum(clinrows)
  lin = 1:nlin
  lin_nnzj = length(clinvals)
  @lencheck lin_nnzj clinrows clincols

  adbackend = ADModelNLSBackend(nvar, F!, nequ, ncon - nlin, c!; x0 = x0, kwargs...)

  nnzh = get_nln_nnzh(adbackend, nvar)

  nln_nnzj = get_nln_nnzj(adbackend, nvar, ncon - nlin)
  nnzj = lin_nnzj + nln_nnzj

  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    nnzh = nnzh,
    name = name,
    lin = lin,
    lin_nnzj = lin_nnzj,
    nln_nnzj = nln_nnzj,
    minimize = minimize,
  )
  nls_nnzj = get_residual_nnzj(adbackend, nvar, nequ)
  nls_nnzh = get_residual_nnzh(adbackend, nvar)
  nls_meta = NLSMeta{T, S}(nequ, nvar, nnzj = nls_nnzj, nnzh = nls_nnzh, lin = linequ)
  return ADNLSModel(meta, nls_meta, NLSCounters(), adbackend, F!, clinrows, clincols, clinvals, c!)
end

function ADNLSModel(
  F,
  x0::S,
  nequ::Integer,
  A::AbstractSparseMatrix{Tv, Ti},
  c,
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Tv, Ti}
  clinrows, clincols, clinvals = findnz(A)
  return ADNLSModel(F, x0, nequ, clinrows, clincols, clinvals, c, lcon, ucon; kwargs...)
end

function ADNLSModel!(
  F!,
  x0::S,
  nequ::Integer,
  A::AbstractSparseMatrix{Tv, Ti},
  c!,
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Tv, Ti}
  clinrows, clincols, clinvals = findnz(A)
  return ADNLSModel!(F!, x0, nequ, clinrows, clincols, clinvals, c!, lcon, ucon; kwargs...)
end

function ADNLSModel!(
  F!,
  x0::S,
  nequ::Integer,
  A::AbstractSparseMatrix{Tv, Ti},
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Tv, Ti}
  clinrows, clincols, clinvals = findnz(A)
  return ADNLSModel!(
    F!,
    x0,
    nequ,
    clinrows,
    clincols,
    clinvals,
    (cx, x) -> cx,
    lcon,
    ucon;
    kwargs...,
  )
end

function ADNLSModel(
  F,
  x0::S,
  nequ::Integer,
  lvar::S,
  uvar::S,
  clinrows::Si,
  clincols::Si,
  clinvals::S,
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Si}
  function F!(output, x)
    Fx = F(x)
    for i = 1:nequ
      output[i] = Fx[i]
    end
    return output
  end
  return ADNLSModel!(F!, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, lcon, ucon; kwargs...)
end

function ADNLSModel!(
  F!,
  x0::S,
  nequ::Integer,
  lvar::S,
  uvar::S,
  clinrows::Si,
  clincols::Si,
  clinvals::S,
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Si}
  return ADNLSModel!(
    F!,
    x0,
    nequ,
    lvar,
    uvar,
    clinrows,
    clincols,
    clinvals,
    (cx, x) -> cx,
    lcon,
    ucon;
    kwargs...,
  )
end

function ADNLSModel(
  F,
  x0::S,
  nequ::Integer,
  lvar::S,
  uvar::S,
  A::AbstractSparseMatrix{Tv, Ti},
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Tv, Ti}
  function F!(output, x)
    Fx = F(x)
    for i = 1:nequ
      output[i] = Fx[i]
    end
    return output
  end
  return ADNLSModel!(F!, x0, nequ, lvar, uvar, A, lcon, ucon; kwargs...)
end

function ADNLSModel!(
  F!,
  x0::S,
  nequ::Integer,
  lvar::S,
  uvar::S,
  A::AbstractSparseMatrix{Tv, Ti},
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Tv, Ti}
  clinrows, clincols, clinvals = findnz(A)
  return ADNLSModel!(F!, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, lcon, ucon; kwargs...)
end

function ADNLSModel(
  F,
  x0::S,
  nequ::Integer,
  lvar::S,
  uvar::S,
  c,
  lcon::S,
  ucon::S;
  kwargs...,
) where {S}
  function F!(output, x)
    Fx = F(x)
    for i = 1:nequ
      output[i] = Fx[i]
    end
    return output
  end

  function c!(output, x)
    cx = c(x)
    for i = 1:length(cx)
      output[i] = cx[i]
    end
    return output
  end

  return ADNLSModel!(F!, x0, nequ, lvar, uvar, c!, lcon, ucon; kwargs...)
end

function ADNLSModel!(
  F!,
  x0::S,
  nequ::Integer,
  lvar::S,
  uvar::S,
  c!,
  lcon::S,
  ucon::S;
  y0::S = fill!(similar(lcon), zero(eltype(S))),
  linequ::AbstractVector{<:Integer} = Int[],
  name::String = "Generic",
  minimize::Bool = true,
  kwargs...,
) where {S}
  T = eltype(S)
  nvar = length(x0)
  ncon = length(lcon)
  @lencheck nvar lvar uvar
  @lencheck ncon ucon y0

  adbackend = ADModelNLSBackend(nvar, F!, nequ, ncon, c!; x0 = x0, kwargs...)

  nnzh = get_nln_nnzh(adbackend, nvar)
  nnzj = get_nln_nnzj(adbackend, nvar, ncon)

  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    nnzh = nnzh,
    nln_nnzj = nnzj,
    name = name,
    minimize = minimize,
  )
  nls_nnzj = get_residual_nnzj(adbackend, nvar, nequ)
  nls_nnzh = get_residual_nnzh(adbackend, nvar)
  nls_meta = NLSMeta{T, S}(nequ, nvar, nnzj = nls_nnzj, nnzh = nls_nnzh, lin = linequ)
  return ADNLSModel(meta, nls_meta, NLSCounters(), adbackend, F!, c!)
end

function ADNLSModel(
  F,
  x0::S,
  nequ::Integer,
  lvar::S,
  uvar::S,
  clinrows::Si,
  clincols::Si,
  clinvals::S,
  c,
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Si}
  function F!(output, x)
    Fx = F(x)
    for i = 1:nequ
      output[i] = Fx[i]
    end
    return output
  end

  function c!(output, x)
    cx = c(x)
    for i = 1:length(cx)
      output[i] = cx[i]
    end
    return output
  end

  return ADNLSModel!(
    F!,
    x0,
    nequ,
    lvar,
    uvar,
    clinrows,
    clincols,
    clinvals,
    c!,
    lcon,
    ucon;
    kwargs...,
  )
end

function ADNLSModel!(
  F!,
  x0::S,
  nequ::Integer,
  lvar::S,
  uvar::S,
  clinrows::Si,
  clincols::Si,
  clinvals::S,
  c!,
  lcon::S,
  ucon::S;
  y0::S = fill!(similar(lcon), zero(eltype(S))),
  linequ::AbstractVector{<:Integer} = Int[],
  name::String = "Generic",
  minimize::Bool = true,
  kwargs...,
) where {S, Si}
  T = eltype(S)
  nvar = length(x0)
  ncon = length(lcon)
  @lencheck nvar lvar uvar
  @lencheck ncon ucon y0

  nlin = isempty(clinrows) ? 0 : maximum(clinrows)
  lin = 1:nlin
  lin_nnzj = length(clinvals)
  @lencheck lin_nnzj clinrows clincols

  adbackend = ADModelNLSBackend(nvar, F!, nequ, ncon - nlin, c!; x0 = x0, kwargs...)

  nnzh = get_nln_nnzh(adbackend, nvar)

  nln_nnzj = get_nln_nnzj(adbackend, nvar, ncon - nlin)
  nnzj = lin_nnzj + nln_nnzj

  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    name = name,
    lin = lin,
    lin_nnzj = lin_nnzj,
    nln_nnzj = nln_nnzj,
    nnzh = nnzh,
    minimize = minimize,
  )
  nls_nnzj = get_residual_nnzj(adbackend, nvar, nequ)
  nls_nnzh = get_residual_nnzh(adbackend, nvar)
  nls_meta = NLSMeta{T, S}(nequ, nvar, nnzj = nls_nnzj, nnzh = nls_nnzh, lin = linequ)
  return ADNLSModel(meta, nls_meta, NLSCounters(), adbackend, F!, clinrows, clincols, clinvals, c!)
end

function ADNLSModel(
  F,
  x0,
  nequ::Integer,
  lvar::S,
  uvar::S,
  A::AbstractSparseMatrix{Tv, Ti},
  c,
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Tv, Ti}
  clinrows, clincols, clinvals = findnz(A)
  return ADNLSModel(F, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, c, lcon, ucon; kwargs...)
end

function ADNLSModel!(
  F!,
  x0,
  nequ::Integer,
  lvar::S,
  uvar::S,
  A::AbstractSparseMatrix{Tv, Ti},
  c!,
  lcon::S,
  ucon::S;
  kwargs...,
) where {S, Tv, Ti}
  clinrows, clincols, clinvals = findnz(A)
  return ADNLSModel!(
    F!,
    x0,
    nequ,
    lvar,
    uvar,
    clinrows,
    clincols,
    clinvals,
    c!,
    lcon,
    ucon;
    kwargs...,
  )
end

function NLPModels.residual!(nls::ADNLSModel, x::AbstractVector, Fx::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  nls.F!(Fx, x)
  return Fx
end

function NLPModels.jac_structure_residual!(
  nls::ADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nls.nls_meta.nnzj rows cols
  return jac_structure_residual!(nls.adbackend.jacobian_residual_backend, nls, rows, cols)
end

function NLPModels.jac_coord_residual!(nls::ADNLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nnzj vals
  increment!(nls, :neval_jac_residual)
  jac_coord_residual!(nls.adbackend.jacobian_residual_backend, nls, x, vals)
  return vals
end

function NLPModels.jprod_residual!(
  nls::ADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nls.meta.nvar x v
  @lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  F = get_F(nls, nls.adbackend.jprod_residual_backend)
  Jprod!(nls.adbackend.jprod_residual_backend, Jv, F, x, v, Val(:F))
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::ADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.meta.nvar x Jtv
  @lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_jtprod_residual)
  F = get_F(nls, nls.adbackend.jtprod_residual_backend)
  Jtprod!(nls.adbackend.jtprod_residual_backend, Jtv, F, x, v, Val(:F))
  return Jtv
end

#=
function NLPModels.hess_residual(nls::ADNLSModel, x::AbstractVector, v::AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_hess_residual)
  F = get_F(nls, nls.adbackend.hessian_residual_backend)
  ϕ(x) = dot(F(x), v)
  return Symmetric(hessian(nls.adbackend.hessian_residual_backend, ϕ, x), :L)
end
=#

function NLPModels.hess_structure_residual!(
  nls::ADNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nls.nls_meta.nnzh rows cols
  return hess_structure_residual!(nls.adbackend.hessian_residual_backend, nls, rows, cols)
end

function NLPModels.hess_coord_residual!(
  nls::ADNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nequ v
  @lencheck nls.nls_meta.nnzh vals
  increment!(nls, :neval_hess_residual)
  return hess_coord_residual!(nls.adbackend.hessian_residual_backend, nls, x, v, vals)
end

function NLPModels.hprod_residual!(
  nls::ADNLSModel,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  @lencheck nls.meta.nvar x v Hiv
  increment!(nls, :neval_hprod_residual)
  hprod_residual!(nls.adbackend.hprod_residual_backend, nls, x, v, i, Hiv)
  return Hiv
end
