# Tutorial

```@contents
Pages = ["tutorial.md"]
```

## ADNLPModel Tutorial

ADNLPModel is simple to use and is useful for classrooms.
It only needs the objective function ``f`` and a starting point ``x^0`` to be
well-defined.
For constrained problems, you'll also need the constraints function ``c``, and
the constraints vectors ``c_L`` and ``c_U``, such that ``c_L \leq c(x) \leq c_U``.
Equality constraints will be automatically identified as those indices ``i`` for
which ``c_{L_i} = c_{U_i}``.

Let's define the famous Rosenbrock function
```math
f(x) = (x_1 - 1)^2 + 100(x_2 - x_1^2)^2,
```
with starting point ``x^0 = (-1.2,1.0)``.

```@example adnlp
using ADNLPModels

nlp = ADNLPModel(x->(x[1] - 1.0)^2 + 100*(x[2] - x[1]^2)^2 , [-1.2; 1.0])
```

This is enough to define the model.
Let's get the objective function value at ``x^0``, using only `nlp`.

```@example adnlp
using NLPModels # To access the API

fx = obj(nlp, nlp.meta.x0)
println("fx = $fx")
```

Done.
Let's try the gradient and Hessian.

```@example adnlp
gx = grad(nlp, nlp.meta.x0)
Hx = hess(nlp, nlp.meta.x0)
println("gx = $gx")
println("Hx = $Hx")
```

Notice that the Hessian is *dense*. This is a current limitation of this model. It
doesn't return sparse matrices, so use it with care.

Let's do something a little more complex here, defining a function to try to
solve this problem through steepest descent method with Armijo search.
Namely, the method

1. Given ``x^0``, ``\varepsilon > 0``, and ``\eta \in (0,1)``. Set ``k = 0``;
2. If ``\Vert \nabla f(x^k) \Vert < \varepsilon`` STOP with ``x^* = x^k``;
3. Compute ``d^k = -\nabla f(x^k)``;
4. Compute ``\alpha_k \in (0,1]`` such that ``f(x^k + \alpha_kd^k) < f(x^k) + \alpha_k\eta \nabla f(x^k)^Td^k``
5. Define ``x^{k+1} = x^k + \alpha_kx^k``
6. Update ``k = k + 1`` and go to step 2.

```@example adnlp
using LinearAlgebra

function steepest(nlp; itmax=100000, eta=1e-4, eps=1e-6, sigma=0.66)
  x = nlp.meta.x0
  fx = obj(nlp, x)
  ∇fx = grad(nlp, x)
  slope = dot(∇fx, ∇fx)
  ∇f_norm = sqrt(slope)
  iter = 0
  while ∇f_norm > eps && iter < itmax
    t = 1.0
    x_trial = x - t * ∇fx
    f_trial = obj(nlp, x_trial)
    while f_trial > fx - eta * t * slope
      t *= sigma
      x_trial = x - t * ∇fx
      f_trial = obj(nlp, x_trial)
    end
    x = x_trial
    fx = f_trial
    ∇fx = grad(nlp, x)
    slope = dot(∇fx, ∇fx)
    ∇f_norm = sqrt(slope)
    iter += 1
  end
  optimal = ∇f_norm <= eps
  return x, fx, ∇f_norm, optimal, iter
end

x, fx, ngx, optimal, iter = steepest(nlp)
println("x = $x")
println("fx = $fx")
println("ngx = $ngx")
println("optimal = $optimal")
println("iter = $iter")
```

Maybe this code is too complicated? If you're in a class you just want to show a
Newton step.

```@example adnlp
g(x) = grad(nlp, x)
H(x) = hess(nlp, x)
x = nlp.meta.x0
d = -H(x)\g(x)
```

or a few

```@example adnlp
for i = 1:5
  global x
  x = x - H(x)\g(x)
  println("x = $x")
end
```

Also, notice how we can reuse the method.

```@example adnlp
f(x) = (x[1]^2 + x[2]^2 - 5)^2 + (x[1]*x[2] - 2)^2
x0 = [3.0; 2.0]
nlp = ADNLPModel(f, x0)

x, fx, ngx, optimal, iter = steepest(nlp)
```

External models can be tested with `steepest` as well, as long as they implement `obj` and `grad`.

For constrained minimization, you need the constraints vector and bounds too.
Bounds on the variables can be passed through a new vector.

```@example adnlp2
using NLPModels, ADNLPModels # hide
f(x) = (x[1] - 1.0)^2 + 100*(x[2] - x[1]^2)^2
x0 = [-1.2; 1.0]
lvar = [-Inf; 0.1]
uvar = [0.5; 0.5]
c(x) = [x[1] + x[2] - 2; x[1]^2 + x[2]^2]
lcon = [0.0; -Inf]
ucon = [Inf; 1.0]
nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)

println("cx = $(cons(nlp, nlp.meta.x0))")
println("Jx = $(jac(nlp, nlp.meta.x0))")
```

## ADNLSModel tutorial

In addition to the general nonlinear model, we can define the residual function for a
nonlinear least-squares problem. In other words, the objective function of the problem
is of the form ``f(x) = \tfrac{1}{2}\|F(x)\|^2``, and we can define the function ``F``
and its derivatives.

A simple way to define an NLS problem is with `ADNLSModel`, which uses automatic
differentiation.

```@example nls
using NLPModels, ADNLPModels # hide
F(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]
x0 = [-1.2; 1.0]
nls = ADNLSModel(F, x0, 2) # 2 nonlinear equations
```

```@example nls
residual(nls, x0)
```

```@example nls
jac_residual(nls, x0)
```
