#=
https://nbviewer.jupyter.org/urls/laurentlessard.com/teaching/cs524/examples/Polygon.ipynb
Largest inscribed polygon
T.M.: origin problem is a Max.
=#

function polygon3(args...; n = 100, kwargs...)
  m = Model()
  N = div(n, 2)
  @variable(m, x[1:N])
  @variable(m, y[1:N])

  for i = 1:N
    @NLconstraint(m, x[i]^2 + y[i]^2 <= 1)
  end

  # add ordering constraint to the vertices
  for i = 1:N-1
    @constraint(m, x[i] * y[i+1] - y[i] * x[i+1] >= 0)
  end
  @NLconstraint(m, x[N] * y[1] - y[N] * x[1] >= 0)

  @NLobjective(
    m,
    Min,
    -0.5 * sum(x[i] * y[i+1] - y[i] * x[i+1] for i = 1:N-1) -
    0.5 * (x[N] * y[1] - y[N] * x[1])
  )
  return m
end

function polygon3_autodiff(
  args...;
  n::Int = 200,
  type::Val{T} = Val(Float64),
  adbackend = ADNLPModels.ForwardDiffAD(),
  kwargs...,
) where {T}
  N = div(n, 2)
  function f(y)
    x, y = y[1:N], y[N+1:end]
    return -0.5 * sum(x[i] * y[i+1] - y[i] * x[i+1] for i = 1:N-1) -
            0.5 * (x[N] * y[1] - y[N] * x[1])
  end
  function c(y)
    x, y = y[1:N], y[N+1:end]
    return vcat(
      [x[i]^2 + y[i]^2 for i = 1:N],
      [x[i] * y[i+1] - y[i] * x[i+1] for i = 1:N-1],
      x[N] * y[1] - y[N] * x[1],
    )
  end
  xi = zeros(T, 2 * N)
  return ADNLPModel(
    f,
    xi,
    c,
    vcat(-Inf * ones(T, N), zeros(T, N)),
    vcat(ones(T, N), Inf * ones(T, N)),
    adbackend = adbackend,
    name = "polygon3_autodiff",
  )
end
