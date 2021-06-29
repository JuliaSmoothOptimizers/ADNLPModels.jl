#=
The clnlbeam problem
https://jump.dev/JuMP.jl/stable/tutorials/Nonlinear%20programs/clnlbeam/
Based on an AMPL model by Hande Y. Benson
Copyright (C) 2001 Princeton University All Rights Reserved
Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted, provided that the above copyright notice appear in all copies and that the copyright notice and this permission notice appear in all supporting documentation.
Source: H. Maurer and H.D. Mittelman, "The non-linear beam via optimal control with bound state variables", Optimal Control Applications and Methods 12, pp. 19-31, 1991.
=#

function clnlbeam(args...; n = 300, kwargs...)
  N = div(n - 3, 3)
  h = 1 / N
  alpha = 350
  model = Model()
  @variables(model, begin
    -1 <= t[1:(N+1)] <= 1
    -0.05 <= x[1:(N+1)] <= 0.05
    u[1:(N+1)]
  end)
  @NLobjective(
    model,
    Min,
    sum(
      0.5 * h * (u[i+1]^2 + u[i]^2) + 0.5 * alpha * h * (cos(t[i+1]) + cos(t[i])) for
      i = 1:N
    ),
  )
  @NLconstraint(
    model,
    [i = 1:N],
    x[i+1] - x[i] - 0.5 * h * (sin(t[i+1]) + sin(t[i])) == 0,
  )
  @constraint(model, [i = 1:N], t[i+1] - t[i] - 0.5 * h * u[i+1] - 0.5 * h * u[i] == 0,)

  return model
end

function clnlbeam_autodiff(
  args...;
  n::Int = 300,
  type::Val{T} = Val(Float64),
  adbackend = ADNLPModels.ForwardDiffAD(),
  kwargs...,
) where {T}
  N = div(n - 3, 3)
  h = 1 / N
  alpha = 350
  function f(y)
    t, x, u = y[1:N+1], y[N+2:2*N+2], y[2*N+3:end]
    return sum(
      0.5 * h * (u[i+1]^2 + u[i]^2) + 0.5 * alpha * h * (cos(t[i+1]) + cos(t[i])) for
      i = 1:N
    )
  end
  function c(y)
    t, x, u = y[1:N+1], y[N+2:2*N+2], y[2*N+3:end]
    return vcat(
      [x[i+1] - x[i] - 0.5 * h * (sin(t[i+1]) + sin(t[i])) for i = 1:N],
      [t[i+1] - t[i] - 0.5 * h * u[i+1] - 0.5 * h * u[i] for i = 1:N],
    )
  end
  lvar = vcat(-ones(T, N + 1), -0.05 * ones(T, N + 1), -Inf * ones(T, N + 1))
  uvar = vcat(ones(T, N + 1), 0.05 * ones(T, N + 1), Inf * ones(T, N + 1))
  xi = zeros(T, 3 * N + 3)
  return ADNLPModel(
    f,
    xi,
    lvar,
    uvar,
    c,
    zeros(T, 2 * N),
    zeros(T, 2 * N),
    adbackend = adbackend,
    name = "clnlbeam_autodiff",
  )
end
