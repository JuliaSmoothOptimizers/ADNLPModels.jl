var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [ADNLPModels]","category":"page"},{"location":"reference/#ADNLPModels.ADNLPModel-Union{Tuple{S}, Tuple{Any, S}} where S","page":"Reference","title":"ADNLPModels.ADNLPModel","text":"ADNLPModel(f, x0)\nADNLPModel(f, x0, lvar, uvar)\nADNLPModel(f, x0, c, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)\n\nADNLPModel is an AbstractNLPModel using automatic differentiation to compute the derivatives. The problem is defined as\n\n min  f(x)\ns.to  lcon ≤ c(x) ≤ ucon\n      lvar ≤   x  ≤ uvar.\n\nThe following keyword arguments are available to all constructors:\n\nname: The name of the model (default: \"Generic\")\n\nThe following keyword arguments are available to the constructors for constrained problems:\n\nlin: An array of indexes of the linear constraints (default: Int[])\ny0: An inital estimate to the Lagrangian multipliers (default: zeros)\n\n\n\n\n\n","category":"method"},{"location":"reference/#ADNLPModels.ADNLSModel-Union{Tuple{S}, Tuple{Any, S, Any}} where S","page":"Reference","title":"ADNLPModels.ADNLSModel","text":"ADNLSModel(F, x0, nequ)\nADNLSModel(F, x0, nequ, lvar, uvar)\nADNLSModel(F, x0, nequ, c, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, c, lcon, ucon)\n\nADNLSModel is an Nonlinear Least Squares model using ForwardDiff to compute the derivatives. The problem is defined as\n\n min  ½‖F(x)‖²\ns.to  lcon ≤ c(x) ≤ ucon\n      lvar ≤   x  ≤ uvar\n\nThe following keyword arguments are available to all constructors:\n\nlinequ: An array of indexes of the linear equations (default: Int[])\nname: The name of the model (default: \"Generic\")\n\nThe following keyword arguments are available to the constructors for constrained problems:\n\nlin: An array of indexes of the linear constraints (default: Int[])\ny0: An inital estimate to the Lagrangian multipliers (default: zeros)\n\n\n\n\n\n","category":"method"},{"location":"#ADNLPModelss","page":"Home","title":"ADNLPModelss","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides AD-based model implementations that conform to the NLPModels API. The following packages are supported:","category":"page"},{"location":"","page":"Home","title":"Home","text":"ForwardDiff.jl: default choice.\nZygote.jl: you must load Zygote.jl separately and pass ADNLPModels.ZygoteAD() as the adbackend keyword argument to the ADNLPModel or ADNLSModel constructor.\nReverseDiff.jl: you must load ReverseDiff.jl separately and pass ADNLPModels.ReverseDiffAD() as the adbackend keyword argument to the ADNLPModel or ADNLSModel constructor.","category":"page"},{"location":"#Install","page":"Home","title":"Install","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Install ADNLPModels.jl with the following command.","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add ADNLPModels","category":"page"},{"location":"#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package defines two models, ADNLPModel for general nonlinear optimization, and ADNLSModel other for nonlinear least-squares problems.","category":"page"},{"location":"","page":"Home","title":"Home","text":"ADNLPModel\nADNLSModel","category":"page"},{"location":"#ADNLPModels.ADNLPModel","page":"Home","title":"ADNLPModels.ADNLPModel","text":"ADNLPModel(f, x0)\nADNLPModel(f, x0, lvar, uvar)\nADNLPModel(f, x0, c, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)\n\nADNLPModel is an AbstractNLPModel using automatic differentiation to compute the derivatives. The problem is defined as\n\n min  f(x)\ns.to  lcon ≤ c(x) ≤ ucon\n      lvar ≤   x  ≤ uvar.\n\nThe following keyword arguments are available to all constructors:\n\nname: The name of the model (default: \"Generic\")\n\nThe following keyword arguments are available to the constructors for constrained problems:\n\nlin: An array of indexes of the linear constraints (default: Int[])\ny0: An inital estimate to the Lagrangian multipliers (default: zeros)\n\n\n\n\n\n","category":"type"},{"location":"#ADNLPModels.ADNLSModel","page":"Home","title":"ADNLPModels.ADNLSModel","text":"ADNLSModel(F, x0, nequ)\nADNLSModel(F, x0, nequ, lvar, uvar)\nADNLSModel(F, x0, nequ, c, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, c, lcon, ucon)\n\nADNLSModel is an Nonlinear Least Squares model using ForwardDiff to compute the derivatives. The problem is defined as\n\n min  ½‖F(x)‖²\ns.to  lcon ≤ c(x) ≤ ucon\n      lvar ≤   x  ≤ uvar\n\nThe following keyword arguments are available to all constructors:\n\nlinequ: An array of indexes of the linear equations (default: Int[])\nname: The name of the model (default: \"Generic\")\n\nThe following keyword arguments are available to the constructors for constrained problems:\n\nlin: An array of indexes of the linear constraints (default: Int[])\ny0: An inital estimate to the Lagrangian multipliers (default: zeros)\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"Check the Tutorial for more details on the usage.","category":"page"},{"location":"#License","page":"Home","title":"License","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This content is released under the MPL2.0 License.","category":"page"},{"location":"#Contents","page":"Home","title":"Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"tutorial/#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Pages = [\"tutorial.md\"]","category":"page"},{"location":"tutorial/#ADNLPModel-Tutorial","page":"Tutorial","title":"ADNLPModel Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"ADNLPModel is simple to use and is useful for classrooms. It only needs the objective function f and a starting point x^0 to be well-defined. For constrained problems, you'll also need the constraints function c, and the constraints vectors c_L and c_U, such that c_L leq c(x) leq c_U. Equality constraints will be automatically identified as those indices i for which c_L_i = c_U_i.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Let's define the famous Rosenbrock function","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"f(x) = (x_1 - 1)^2 + 100(x_2 - x_1^2)^2","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"with starting point x^0 = (-1210).","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using ADNLPModels\n\nnlp = ADNLPModel(x->(x[1] - 1.0)^2 + 100*(x[2] - x[1]^2)^2 , [-1.2; 1.0])","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"This is enough to define the model. Let's get the objective function value at x^0, using only nlp.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using NLPModels # To access the API\n\nfx = obj(nlp, nlp.meta.x0)\nprintln(\"fx = $fx\")","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Done. Let's try the gradient and Hessian.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"gx = grad(nlp, nlp.meta.x0)\nHx = hess(nlp, nlp.meta.x0)\nprintln(\"gx = $gx\")\nprintln(\"Hx = $Hx\")","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Notice that the Hessian is dense. This is a current limitation of this model. It doesn't return sparse matrices, so use it with care.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Let's do something a little more complex here, defining a function to try to solve this problem through steepest descent method with Armijo search. Namely, the method","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Given x^0, varepsilon  0, and eta in (01). Set k = 0;\nIf Vert nabla f(x^k) Vert  varepsilon STOP with x^* = x^k;\nCompute d^k = -nabla f(x^k);\nCompute alpha_k in (01 such that f(x^k + alpha_kd^k)  f(x^k) + alpha_keta nabla f(x^k)^Td^k\nDefine x^k+1 = x^k + alpha_kx^k\nUpdate k = k + 1 and go to step 2.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using LinearAlgebra\n\nfunction steepest(nlp; itmax=100000, eta=1e-4, eps=1e-6, sigma=0.66)\n  x = nlp.meta.x0\n  fx = obj(nlp, x)\n  ∇fx = grad(nlp, x)\n  slope = dot(∇fx, ∇fx)\n  ∇f_norm = sqrt(slope)\n  iter = 0\n  while ∇f_norm > eps && iter < itmax\n    t = 1.0\n    x_trial = x - t * ∇fx\n    f_trial = obj(nlp, x_trial)\n    while f_trial > fx - eta * t * slope\n      t *= sigma\n      x_trial = x - t * ∇fx\n      f_trial = obj(nlp, x_trial)\n    end\n    x = x_trial\n    fx = f_trial\n    ∇fx = grad(nlp, x)\n    slope = dot(∇fx, ∇fx)\n    ∇f_norm = sqrt(slope)\n    iter += 1\n  end\n  optimal = ∇f_norm <= eps\n  return x, fx, ∇f_norm, optimal, iter\nend\n\nx, fx, ngx, optimal, iter = steepest(nlp)\nprintln(\"x = $x\")\nprintln(\"fx = $fx\")\nprintln(\"ngx = $ngx\")\nprintln(\"optimal = $optimal\")\nprintln(\"iter = $iter\")","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Maybe this code is too complicated? If you're in a class you just want to show a Newton step.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"g(x) = grad(nlp, x)\nH(x) = hess(nlp, x)\nx = nlp.meta.x0\nd = -H(x)\\g(x)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"or a few","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"for i = 1:5\n  global x\n  x = x - H(x)\\g(x)\n  println(\"x = $x\")\nend","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Also, notice how we can reuse the method.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"f(x) = (x[1]^2 + x[2]^2 - 5)^2 + (x[1]*x[2] - 2)^2\nx0 = [3.0; 2.0]\nnlp = ADNLPModel(f, x0)\n\nx, fx, ngx, optimal, iter = steepest(nlp)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"External models can be tested with steepest as well, as long as they implement obj and grad.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"For constrained minimization, you need the constraints vector and bounds too. Bounds on the variables can be passed through a new vector.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using NLPModels, ADNLPModels # hide\nf(x) = (x[1] - 1.0)^2 + 100*(x[2] - x[1]^2)^2\nx0 = [-1.2; 1.0]\nlvar = [-Inf; 0.1]\nuvar = [0.5; 0.5]\nc(x) = [x[1] + x[2] - 2; x[1]^2 + x[2]^2]\nlcon = [0.0; -Inf]\nucon = [Inf; 1.0]\nnlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)\n\nprintln(\"cx = $(cons(nlp, nlp.meta.x0))\")\nprintln(\"Jx = $(jac(nlp, nlp.meta.x0))\")","category":"page"},{"location":"tutorial/#ADNLSModel-tutorial","page":"Tutorial","title":"ADNLSModel tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"In addition to the general nonlinear model, we can define the residual function for a nonlinear least-squares problem. In other words, the objective function of the problem is of the form f(x) = tfrac12F(x)^2, and we can define the function F and its derivatives.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"A simple way to define an NLS problem is with ADNLSModel, which uses automatic differentiation.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using NLPModels, ADNLPModels # hide\nF(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]\nx0 = [-1.2; 1.0]\nnls = ADNLSModel(F, x0, 2) # 2 nonlinear equations","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"residual(nls, x0)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"jac_residual(nls, x0)","category":"page"}]
}
