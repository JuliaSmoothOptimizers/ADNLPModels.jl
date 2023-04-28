var documenterSearchIndex = {"docs":
[{"location":"backend/#How-to-switch-backend-in-ADNLPModels","page":"Backend","title":"How to switch backend in ADNLPModels","text":"","category":"section"},{"location":"backend/","page":"Backend","title":"Backend","text":"ADNLPModels allows the use of different backends to compute the derivatives required within NLPModel API. It uses ForwardDiff.jl, ReverseDiff.jl, and Zygote.jl via optional depencies.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"The backend information is in a structure ADNLPModels.ADModelBackend in the attribute adbackend of a ADNLPModel, it can also be accessed with get_adbackend.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"The functions used internally to define the NLPModel API and the possible backends are defined in the following table:","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"Functions FowardDiff backends ReverseDiff backends Zygote backends\ngradient and gradient! ForwardDiffADGradient ReverseDiffADGradient ZygoteADGradient\njacobian ForwardDiffADJacobian ReverseDiffADJacobian ZygoteADJacobian\nhessian ForwardDiffADHessian ReverseDiffADHessian ZygoteADHessian\nJprod ForwardDiffADJprod ReverseDiffADJprod ZygoteADJprod\nJtprod ForwardDiffADJtprod ReverseDiffADJtprod ZygoteADJtprod\nHvprod ForwardDiffADHvprod ReverseDiffADHvprod –\ndirectional_second_derivative ForwardDiffADGHjvprod – –","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"The functions hess_structure!, hess_coord!, jac_structure! and jac_coord! defined in ad.jl are generic to all the backends for now.","category":"page"},{"location":"backend/#Examples","page":"Backend","title":"Examples","text":"","category":"section"},{"location":"backend/","page":"Backend","title":"Backend","text":"We now present a serie of practical examples. For simplicity, we focus here on unconstrained optimization problem. All these examples can be generalized to problems with bounds, constraints or nonlinear least-squares.","category":"page"},{"location":"backend/#Use-another-backend","page":"Backend","title":"Use another backend","text":"","category":"section"},{"location":"backend/","page":"Backend","title":"Backend","text":"As shown in Tutorial, it is very straightforward to instantiate an ADNLPModel using an objective function and an initial guess.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"using ADNLPModels, NLPModels\nf(x) = sum(x)\nx0 = ones(3)\nnlp = ADNLPModel(f, x0)\ngrad(nlp, nlp.meta.x0) # returns the gradient at x0","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"Thanks to the backends inside ADNLPModels.jl, it is easy to change the backend for one (or more) function using the kwargs presented in the table above.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"nlp = ADNLPModel(f, x0, gradient_backend = ADNLPModels.ReverseDiffADGradient)\ngrad(nlp, nlp.meta.x0) # returns the gradient at x0 using `ReverseDiff`","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"It is also possible to try some new implementation for each function. First, we define a new ADBackend structure.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"struct NewADGradient <: ADNLPModels.ADBackend end\nfunction NewADGradient(\n  nvar::Integer,\n  f,\n  ncon::Integer = 0,\n  c::Function = (args...) -> [];\n  kwargs...,\n)\n  return NewADGradient()\nend","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"Then, we implement the desired functions following the table above.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"ADNLPModels.gradient(adbackend::NewADGradient, f, x) = rand(Float64, size(x))\nfunction ADNLPModels.gradient!(adbackend::NewADGradient, g, f, x)\n  g .= rand(Float64, size(x))\n  return g\nend","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"Finally, we use the homemade backend to compute the gradient.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"nlp = ADNLPModel(sum, ones(3), gradient_backend = NewADGradient)\ngrad(nlp, nlp.meta.x0) # returns the gradient at x0 using `NewADGradient`","category":"page"},{"location":"backend/#Change-backend","page":"Backend","title":"Change backend","text":"","category":"section"},{"location":"backend/","page":"Backend","title":"Backend","text":"Once an instance of an ADNLPModel has been created, it is possible to change the backends without re-instantiating the model.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"using ADNLPModels, NLPModels\nf(x) = 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2\nx0 = 3 * ones(2)\nnlp = ADNLPModel(f, x0)\nget_adbackend(nlp) # returns the `ADModelBackend` structure that regroup all the various backends.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"There are currently two ways to modify instantiated backends. The first one is to instantiate a new ADModelBackend and use set_adbackend! to modify nlp.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"adback = ADNLPModels.ADModelBackend(nlp.meta.nvar, nlp.f, gradient_backend = ADNLPModels.ForwardDiffADGradient)\nset_adbackend!(nlp, adback)\nget_adbackend(nlp)","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"The alternative is to use `set_adbackend! and pass the new backends via kwargs. In the second approach, it is possible to pass either the type of the desired backend or an instance as shown below.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"set_adbackend!(\n  nlp,\n  gradient_backend = ADNLPModels.ForwardDiffADGradient,\n  jtprod_backend = ADNLPModels.ForwardDiffADJtprod(),\n)\nget_adbackend(nlp)","category":"page"},{"location":"backend/#Support-multiple-precision-without-having-to-recreate-the-model","page":"Backend","title":"Support multiple precision without having to recreate the model","text":"","category":"section"},{"location":"backend/","page":"Backend","title":"Backend","text":"One of the strength of ADNLPModels.jl is the type flexibility. Let's assume, we first instantiate an ADNLPModel with a Float64 initial guess.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"using ADNLPModels, NLPModels\nf(x) = 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2\nx0 = 3 * ones(2) # Float64 initial guess\nnlp = ADNLPModel(f, x0)","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"Then, the gradient will return a vector of Float64.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"x64 = rand(2)\ngrad(nlp, x64)","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"It is now possible to move to a different type, for instance Float32, while keeping the instance nlp.","category":"page"},{"location":"backend/","page":"Backend","title":"Backend","text":"x0_32 = ones(Float32, 2)\nset_adbackend!(nlp, gradient_backend = ADNLPModels.ForwardDiffADGradient, x0 = x0_32)\nx32 = rand(Float32, 2)\ngrad(nlp, x32)","category":"page"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [ADNLPModels]","category":"page"},{"location":"reference/#ADNLPModels.ADModelBackend","page":"Reference","title":"ADNLPModels.ADModelBackend","text":"ADModelBackend(gradient_backend, hprod_backend, jprod_backend, jtprod_backend, jacobian_backend, hessian_backend, ghjvprod_backend, hprod_residual_backend, jprod_residual_backend, jtprod_residual_backend, jacobian_residual_backend, hessian_residual_backend)\n\nStructure that define the different backend used to compute automatic differentiation of an ADNLPModel/ADNLSModel model. The different backend are all subtype of ADBackend and are respectively used for:\n\ngradient computation;\nhessian-vector products;\njacobian-vector products;\ntranspose jacobian-vector products;\njacobian computation;\nhessian computation;\ndirectional second derivative computation, i.e. gᵀ ∇²cᵢ(x) v.\n\nThe default constructors are      ADModelBackend(nvar, f, ncon = 0, c::Function = (args...) -> []; kwargs...)     ADModelNLSBackend(nvar, F!, nequ, ncon = 0, c::Function = (args...) -> []; kwargs...)\n\nwhere the kwargs are either the different backends as listed below or arguments passed to the backend's constructors:\n\ngradient_backend = ForwardDiffADGradient;\nhprod_backend = ForwardDiffADHvprod;\njprod_backend = ForwardDiffADJprod;\njtprod_backend = ForwardDiffADJtprod;\njacobian_backend = SparseForwardADJacobian;\nhessian_backend = ForwardDiffADHessian;\nghjvprod_backend = ForwardDiffADGHjvprod;\nhprod_residual_backend = ForwardDiffADHvprod for ADNLSModel and EmptyADbackend otherwise;\njprod_residual_backend = ForwardDiffADJprod for ADNLSModel and EmptyADbackend otherwise;\njtprod_residual_backend = ForwardDiffADJtprod for ADNLSModel and EmptyADbackend otherwise;\njacobian_residual_backend = SparseForwardADJacobian for ADNLSModel and EmptyADbackend otherwise;\nhessian_residual_backend = ForwardDiffADHessian for ADNLSModel and EmptyADbackend otherwise.\n\n\n\n\n\n","category":"type"},{"location":"reference/#ADNLPModels.ADNLPModel-Union{Tuple{S}, Tuple{Any, S}} where S","page":"Reference","title":"ADNLPModels.ADNLPModel","text":"ADNLPModel(f, x0)\nADNLPModel(f, x0, lvar, uvar)\nADNLPModel(f, x0, clinrows, clincols, clinvals, lcon, ucon)\nADNLPModel(f, x0, A, lcon, ucon)\nADNLPModel(f, x0, c, lcon, ucon)\nADNLPModel(f, x0, clinrows, clincols, clinvals, c, lcon, ucon)\nADNLPModel(f, x0, A, c, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, clinrows, clincols, clinvals, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, A, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, clinrows, clincols, clinvals, c, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, A, c, lcon, ucon)\n\nADNLPModel is an AbstractNLPModel using automatic differentiation to compute the derivatives. The problem is defined as\n\n min  f(x)\ns.to  lcon ≤ (  Ax  ) ≤ ucon\n             ( c(x) )\n      lvar ≤   x  ≤ uvar.\n\nThe following keyword arguments are available to all constructors:\n\nminimize: A boolean indicating whether this is a minimization problem (default: true)\nname: The name of the model (default: \"Generic\")\n\nThe following keyword arguments are available to the constructors for constrained problems:\n\ny0: An inital estimate to the Lagrangian multipliers (default: zeros)\n\nADNLPModel uses ForwardDiff and ReverseDiff for the automatic differentiation. One can specify a new backend with the keyword arguments backend::ADNLPModels.ADBackend. There are three pre-coded backends:\n\nthe default ForwardDiffAD.\nReverseDiffAD.\nZygoteDiffAD accessible after loading Zygote.jl in your environment.\n\nFor an advanced usage, one can define its own backend and redefine the API as done in ADNLPModels.jl/src/forward.jl.\n\nExamples\n\nusing ADNLPModels\nf(x) = sum(x)\nx0 = ones(3)\nnvar = 3\nADNLPModel(f, x0) # uses the default ForwardDiffAD backend.\nADNLPModel(f, x0; backend = ADNLPModels.ReverseDiffAD) # uses ReverseDiffAD backend.\n\nusing Zygote\nADNLPModel(f, x0; backend = ADNLPModels.ZygoteAD)\n\nusing ADNLPModels\nf(x) = sum(x)\nx0 = ones(3)\nc(x) = [1x[1] + x[2]; x[2]]\nnvar, ncon = 3, 2\nADNLPModel(f, x0, c, zeros(ncon), zeros(ncon)) # uses the default ForwardDiffAD backend.\nADNLPModel(f, x0, c, zeros(ncon), zeros(ncon); backend = ADNLPModels.ReverseDiffAD) # uses ReverseDiffAD backend.\n\nusing Zygote\nADNLPModel(f, x0, c, zeros(ncon), zeros(ncon); backend = ADNLPModels.ZygoteAD)\n\nFor in-place constraints function, use one of the following constructors:\n\nADNLPModel!(f, x0, c!, lcon, ucon)\nADNLPModel!(f, x0, clinrows, clincols, clinvals, c!, lcon, ucon)\nADNLPModel!(f, x0, A, c!, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, c!, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, clinrows, clincols, clinvals, c!, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, A, c!, lcon, ucon)\n\nwhere the constraint function has the signature c!(output, input).\n\nusing ADNLPModels\nf(x) = sum(x)\nx0 = ones(3)\nfunction c!(output, x) \n  output[1] = 1x[1] + x[2]\n  output[2] = x[2]\nend\nnvar, ncon = 3, 2\nnlp = ADNLPModel!(f, x0, c!, zeros(ncon), zeros(ncon)) # uses the default ForwardDiffAD backend.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ADNLPModels.ADNLSModel-Union{Tuple{S}, Tuple{Any, S, Integer}} where S","page":"Reference","title":"ADNLPModels.ADNLSModel","text":"ADNLSModel(F, x0, nequ)\nADNLSModel(F, x0, nequ, lvar, uvar)\nADNLSModel(F, x0, nequ, clinrows, clincols, clinvals, lcon, ucon)\nADNLSModel(F, x0, nequ, A, lcon, ucon)\nADNLSModel(F, x0, nequ, c, lcon, ucon)\nADNLSModel(F, x0, nequ, clinrows, clincols, clinvals, c, lcon, ucon)\nADNLSModel(F, x0, nequ, A, c, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, A, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, c, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, c, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, A, c, lcon, ucon)\n\nADNLSModel is an Nonlinear Least Squares model using automatic differentiation to compute the derivatives. The problem is defined as\n\n min  ½‖F(x)‖²\ns.to  lcon ≤ (  Ax  ) ≤ ucon\n             ( c(x) )\n      lvar ≤   x  ≤ uvar\n\nwhere nequ is the size of the vector F(x) and the linear constraints come first.\n\nThe following keyword arguments are available to all constructors:\n\nlinequ: An array of indexes of the linear equations (default: Int[])\nminimize: A boolean indicating whether this is a minimization problem (default: true)\nname: The name of the model (default: \"Generic\")\n\nThe following keyword arguments are available to the constructors for constrained problems:\n\ny0: An inital estimate to the Lagrangian multipliers (default: zeros)\n\nADNLSModel uses ForwardDiff and ReverseDiff for the automatic differentiation. One can specify a new backend with the keyword arguments backend::ADNLPModels.ADBackend. There are three pre-coded backends:\n\nthe default ForwardDiffAD.\nReverseDiffAD.\nZygoteDiffAD accessible after loading Zygote.jl in your environment.\n\nFor an advanced usage, one can define its own backend and redefine the API as done in ADNLPModels.jl/src/forward.jl.\n\nExamples\n\nusing ADNLPModels\nF(x) = [x[2]; x[1]]\nnequ = 2\nx0 = ones(3)\nnvar = 3\nADNLSModel(F, x0, nequ) # uses the default ForwardDiffAD backend.\nADNLSModel(F, x0, nequ; backend = ADNLPModels.ReverseDiffAD) # uses ReverseDiffAD backend.\n\nusing Zygote\nADNLSModel(F, x0, nequ; backend = ADNLPModels.ZygoteAD)\n\nusing ADNLPModels\nF(x) = [x[2]; x[1]]\nnequ = 2\nx0 = ones(3)\nc(x) = [1x[1] + x[2]; x[2]]\nnvar, ncon = 3, 2\nADNLSModel(F, x0, nequ, c, zeros(ncon), zeros(ncon)) # uses the default ForwardDiffAD backend.\nADNLSModel(F, x0, nequ, c, zeros(ncon), zeros(ncon); backend = ADNLPModels.ReverseDiffAD) # uses ReverseDiffAD backend.\n\nusing Zygote\nADNLSModel(F, x0, nequ, c, zeros(ncon), zeros(ncon); backend = ADNLPModels.ZygoteAD)\n\nFor in-place constraints and residual function, use one of the following constructors:\n\nADNLSModel!(F!, x0, nequ)\nADNLSModel!(F!, x0, nequ, lvar, uvar)\nADNLSModel!(F!, x0, nequ, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, clinrows, clincols, clinvals, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, clinrows, clincols, clinvals, lcon, ucon)\nADNLSModel!(F!, x0, nequ, A, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, A, lcon, ucon)\nADNLSModel!(F!, x0, nequ, lvar, uvar, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, lcon, ucon)\nADNLSModel!(F!, x0, nequ, lvar, uvar, A, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, lvar, uvar, A, clcon, ucon)\n\nwhere the constraint function has the signature c!(output, input).\n\nusing ADNLPModels\nfunction F!(output, x)\n  output[1] = x[2]\n  output[2] = x[1]\nend\nnequ = 2\nx0 = ones(3)\nfunction c!(output, x) \n  output[1] = 1x[1] + x[2]\n  output[2] = x[2]\nend\nnvar, ncon = 3, 2\nnls = ADNLSModel!(F!, x0, nequ, c!, zeros(ncon), zeros(ncon))\n\n\n\n\n\n","category":"method"},{"location":"reference/#ADNLPModels.get_F-Tuple{ADNLPModels.AbstractADNLSModel}","page":"Reference","title":"ADNLPModels.get_F","text":"get_F(nls)\nget_F(nls, ::ADBackend)\n\nReturn the out-of-place version of nls.F!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ADNLPModels.get_adbackend-Tuple{Union{ADNLPModels.AbstractADNLPModel{T, S}, ADNLPModels.AbstractADNLSModel{T, S}} where {T, S}}","page":"Reference","title":"ADNLPModels.get_adbackend","text":"get_adbackend(nlp)\n\nReturns the value adbackend from nlp.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ADNLPModels.get_c-Tuple{Union{ADNLPModels.AbstractADNLPModel{T, S}, ADNLPModels.AbstractADNLSModel{T, S}} where {T, S}}","page":"Reference","title":"ADNLPModels.get_c","text":"get_c(nlp)\nget_c(nlp, ::ADBackend)\n\nReturn the out-of-place version of nlp.c!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ADNLPModels.get_lag-Tuple{ADNLPModels.AbstractADNLPModel, ADNLPModels.ADBackend, Real}","page":"Reference","title":"ADNLPModels.get_lag","text":"get_lag(nlp, b::ADBackend, obj_weight)\nget_lag(nlp, b::ADBackend, obj_weight, y)\n\nReturn the lagrangian function ℓ(x) = obj_weight * f(x) + c(x)ᵀy.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ADNLPModels.get_nln_nnzh-Tuple{ADNLPModels.ADModelBackend, Any}","page":"Reference","title":"ADNLPModels.get_nln_nnzh","text":"get_nln_nnzh(::ADBackend, nvar)\nget_nln_nnzh(b::ADModelBackend, nvar)\n\nFor a given ADBackend of a problem with nvar variables, return the number of nonzeros in the lower triangle of the Hessian. If b is the ADModelBackend then b.hessian_backend is used.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ADNLPModels.get_nln_nnzj-Tuple{ADNLPModels.ADModelBackend, Any, Any}","page":"Reference","title":"ADNLPModels.get_nln_nnzj","text":"get_nln_nnzj(::ADBackend, nvar, ncon)\nget_nln_nnzj(b::ADModelBackend, nvar, ncon)\n\nFor a given ADBackend of a problem with nvar variables and ncon constraints, return the number of nonzeros in the Jacobian of nonlinear constraints. If b is the ADModelBackend then b.jacobian_backend is used.\n\n\n\n\n\n","category":"method"},{"location":"reference/#ADNLPModels.get_residual_nnzj-Tuple{ADNLPModels.ADModelBackend, Any, Any}","page":"Reference","title":"ADNLPModels.get_residual_nnzj","text":"get_residual_nnzj(b::ADModelBackend, nvar, nequ)\n\nReturn get_nln_nnzj(b.jacobian_residual_backend, nvar, nequ).\n\n\n\n\n\n","category":"method"},{"location":"reference/#ADNLPModels.set_adbackend!-Tuple{Union{ADNLPModels.AbstractADNLPModel{T, S}, ADNLPModels.AbstractADNLSModel{T, S}} where {T, S}, ADNLPModels.ADModelBackend}","page":"Reference","title":"ADNLPModels.set_adbackend!","text":"set_adbackend!(nlp, new_adbackend)\nset_adbackend!(nlp; kwargs...)\n\nReplace the current adbackend value of nlp by new_adbackend or instantiate a new one with kwargs, see ADModelBackend. By default, the setter with kwargs will reuse existing backends.\n\n\n\n\n\n","category":"method"},{"location":"#ADNLPModels","page":"Home","title":"ADNLPModels","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides AD-based model implementations that conform to the NLPModels API. The following packages are supported: ForwardDiff.jl, ReverseDiff.jl, and Zygote.jl.","category":"page"},{"location":"#Install","page":"Home","title":"Install","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Install ADNLPModels.jl with the following command.","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add ADNLPModels","category":"page"},{"location":"#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package defines two models, ADNLPModel for general nonlinear optimization, and ADNLSModel for nonlinear least-squares problems.","category":"page"},{"location":"","page":"Home","title":"Home","text":"ADNLPModel\nADNLSModel","category":"page"},{"location":"#ADNLPModels.ADNLPModel","page":"Home","title":"ADNLPModels.ADNLPModel","text":"ADNLPModel(f, x0)\nADNLPModel(f, x0, lvar, uvar)\nADNLPModel(f, x0, clinrows, clincols, clinvals, lcon, ucon)\nADNLPModel(f, x0, A, lcon, ucon)\nADNLPModel(f, x0, c, lcon, ucon)\nADNLPModel(f, x0, clinrows, clincols, clinvals, c, lcon, ucon)\nADNLPModel(f, x0, A, c, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, clinrows, clincols, clinvals, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, A, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, clinrows, clincols, clinvals, c, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, A, c, lcon, ucon)\n\nADNLPModel is an AbstractNLPModel using automatic differentiation to compute the derivatives. The problem is defined as\n\n min  f(x)\ns.to  lcon ≤ (  Ax  ) ≤ ucon\n             ( c(x) )\n      lvar ≤   x  ≤ uvar.\n\nThe following keyword arguments are available to all constructors:\n\nminimize: A boolean indicating whether this is a minimization problem (default: true)\nname: The name of the model (default: \"Generic\")\n\nThe following keyword arguments are available to the constructors for constrained problems:\n\ny0: An inital estimate to the Lagrangian multipliers (default: zeros)\n\nADNLPModel uses ForwardDiff and ReverseDiff for the automatic differentiation. One can specify a new backend with the keyword arguments backend::ADNLPModels.ADBackend. There are three pre-coded backends:\n\nthe default ForwardDiffAD.\nReverseDiffAD.\nZygoteDiffAD accessible after loading Zygote.jl in your environment.\n\nFor an advanced usage, one can define its own backend and redefine the API as done in ADNLPModels.jl/src/forward.jl.\n\nExamples\n\nusing ADNLPModels\nf(x) = sum(x)\nx0 = ones(3)\nnvar = 3\nADNLPModel(f, x0) # uses the default ForwardDiffAD backend.\nADNLPModel(f, x0; backend = ADNLPModels.ReverseDiffAD) # uses ReverseDiffAD backend.\n\nusing Zygote\nADNLPModel(f, x0; backend = ADNLPModels.ZygoteAD)\n\nusing ADNLPModels\nf(x) = sum(x)\nx0 = ones(3)\nc(x) = [1x[1] + x[2]; x[2]]\nnvar, ncon = 3, 2\nADNLPModel(f, x0, c, zeros(ncon), zeros(ncon)) # uses the default ForwardDiffAD backend.\nADNLPModel(f, x0, c, zeros(ncon), zeros(ncon); backend = ADNLPModels.ReverseDiffAD) # uses ReverseDiffAD backend.\n\nusing Zygote\nADNLPModel(f, x0, c, zeros(ncon), zeros(ncon); backend = ADNLPModels.ZygoteAD)\n\nFor in-place constraints function, use one of the following constructors:\n\nADNLPModel!(f, x0, c!, lcon, ucon)\nADNLPModel!(f, x0, clinrows, clincols, clinvals, c!, lcon, ucon)\nADNLPModel!(f, x0, A, c!, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, c!, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, clinrows, clincols, clinvals, c!, lcon, ucon)\nADNLPModel(f, x0, lvar, uvar, A, c!, lcon, ucon)\n\nwhere the constraint function has the signature c!(output, input).\n\nusing ADNLPModels\nf(x) = sum(x)\nx0 = ones(3)\nfunction c!(output, x) \n  output[1] = 1x[1] + x[2]\n  output[2] = x[2]\nend\nnvar, ncon = 3, 2\nnlp = ADNLPModel!(f, x0, c!, zeros(ncon), zeros(ncon)) # uses the default ForwardDiffAD backend.\n\n\n\n\n\n","category":"type"},{"location":"#ADNLPModels.ADNLSModel","page":"Home","title":"ADNLPModels.ADNLSModel","text":"ADNLSModel(F, x0, nequ)\nADNLSModel(F, x0, nequ, lvar, uvar)\nADNLSModel(F, x0, nequ, clinrows, clincols, clinvals, lcon, ucon)\nADNLSModel(F, x0, nequ, A, lcon, ucon)\nADNLSModel(F, x0, nequ, c, lcon, ucon)\nADNLSModel(F, x0, nequ, clinrows, clincols, clinvals, c, lcon, ucon)\nADNLSModel(F, x0, nequ, A, c, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, A, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, c, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, c, lcon, ucon)\nADNLSModel(F, x0, nequ, lvar, uvar, A, c, lcon, ucon)\n\nADNLSModel is an Nonlinear Least Squares model using automatic differentiation to compute the derivatives. The problem is defined as\n\n min  ½‖F(x)‖²\ns.to  lcon ≤ (  Ax  ) ≤ ucon\n             ( c(x) )\n      lvar ≤   x  ≤ uvar\n\nwhere nequ is the size of the vector F(x) and the linear constraints come first.\n\nThe following keyword arguments are available to all constructors:\n\nlinequ: An array of indexes of the linear equations (default: Int[])\nminimize: A boolean indicating whether this is a minimization problem (default: true)\nname: The name of the model (default: \"Generic\")\n\nThe following keyword arguments are available to the constructors for constrained problems:\n\ny0: An inital estimate to the Lagrangian multipliers (default: zeros)\n\nADNLSModel uses ForwardDiff and ReverseDiff for the automatic differentiation. One can specify a new backend with the keyword arguments backend::ADNLPModels.ADBackend. There are three pre-coded backends:\n\nthe default ForwardDiffAD.\nReverseDiffAD.\nZygoteDiffAD accessible after loading Zygote.jl in your environment.\n\nFor an advanced usage, one can define its own backend and redefine the API as done in ADNLPModels.jl/src/forward.jl.\n\nExamples\n\nusing ADNLPModels\nF(x) = [x[2]; x[1]]\nnequ = 2\nx0 = ones(3)\nnvar = 3\nADNLSModel(F, x0, nequ) # uses the default ForwardDiffAD backend.\nADNLSModel(F, x0, nequ; backend = ADNLPModels.ReverseDiffAD) # uses ReverseDiffAD backend.\n\nusing Zygote\nADNLSModel(F, x0, nequ; backend = ADNLPModels.ZygoteAD)\n\nusing ADNLPModels\nF(x) = [x[2]; x[1]]\nnequ = 2\nx0 = ones(3)\nc(x) = [1x[1] + x[2]; x[2]]\nnvar, ncon = 3, 2\nADNLSModel(F, x0, nequ, c, zeros(ncon), zeros(ncon)) # uses the default ForwardDiffAD backend.\nADNLSModel(F, x0, nequ, c, zeros(ncon), zeros(ncon); backend = ADNLPModels.ReverseDiffAD) # uses ReverseDiffAD backend.\n\nusing Zygote\nADNLSModel(F, x0, nequ, c, zeros(ncon), zeros(ncon); backend = ADNLPModels.ZygoteAD)\n\nFor in-place constraints and residual function, use one of the following constructors:\n\nADNLSModel!(F!, x0, nequ)\nADNLSModel!(F!, x0, nequ, lvar, uvar)\nADNLSModel!(F!, x0, nequ, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, clinrows, clincols, clinvals, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, clinrows, clincols, clinvals, lcon, ucon)\nADNLSModel!(F!, x0, nequ, A, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, A, lcon, ucon)\nADNLSModel!(F!, x0, nequ, lvar, uvar, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, lvar, uvar, clinrows, clincols, clinvals, lcon, ucon)\nADNLSModel!(F!, x0, nequ, lvar, uvar, A, c!, lcon, ucon)\nADNLSModel!(F!, x0, nequ, lvar, uvar, A, clcon, ucon)\n\nwhere the constraint function has the signature c!(output, input).\n\nusing ADNLPModels\nfunction F!(output, x)\n  output[1] = x[2]\n  output[2] = x[1]\nend\nnequ = 2\nx0 = ones(3)\nfunction c!(output, x) \n  output[1] = 1x[1] + x[2]\n  output[2] = x[2]\nend\nnvar, ncon = 3, 2\nnls = ADNLSModel!(F!, x0, nequ, c!, zeros(ncon), zeros(ncon))\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"Check the Tutorial for more details on the usage.","category":"page"},{"location":"#License","page":"Home","title":"License","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This content is released under the MPL2.0 License.","category":"page"},{"location":"#Bug-reports-and-discussions","page":"Home","title":"Bug reports and discussions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you think you found a bug, feel free to open an issue. Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.","category":"page"},{"location":"","page":"Home","title":"Home","text":"If you want to ask a question not suited for a bug report, feel free to start a discussion here. This forum is for general discussion about this repository and the JuliaSmoothOptimizers, so questions about any of our packages are welcome.","category":"page"},{"location":"#Contents","page":"Home","title":"Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"tutorial/#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Pages = [\"tutorial.md\"]","category":"page"},{"location":"tutorial/#ADNLPModel-Tutorial","page":"Tutorial","title":"ADNLPModel Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"ADNLPModel is simple to use and is useful for classrooms. It only needs the objective function f and a starting point x^0 to be well-defined. For constrained problems, you'll also need the constraints function c, and the constraints vectors c_L and c_U, such that c_L leq c(x) leq c_U. Equality constraints will be automatically identified as those indices i for which c_L_i = c_U_i.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Let's define the famous Rosenbrock function","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"f(x) = (x_1 - 1)^2 + 100(x_2 - x_1^2)^2","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"with starting point x^0 = (-1210).","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using ADNLPModels\n\nnlp = ADNLPModel(x->(x[1] - 1.0)^2 + 100*(x[2] - x[1]^2)^2 , [-1.2; 1.0])","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"This is enough to define the model. Let's get the objective function value at x^0, using only nlp.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using NLPModels # To access the API\n\nfx = obj(nlp, nlp.meta.x0)\nprintln(\"fx = $fx\")","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Done. Let's try the gradient and Hessian.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"gx = grad(nlp, nlp.meta.x0)\nHx = hess(nlp, nlp.meta.x0)\nprintln(\"gx = $gx\")\nprintln(\"Hx = $Hx\")","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Notice that the Hessian is dense. This is a current limitation of this model. It doesn't return sparse matrices, so use it with care.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Let's do something a little more complex here, defining a function to try to solve this problem through steepest descent method with Armijo search. Namely, the method","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Given x^0, varepsilon  0, and eta in (01). Set k = 0;\nIf Vert nabla f(x^k) Vert  varepsilon STOP with x^* = x^k;\nCompute d^k = -nabla f(x^k);\nCompute alpha_k in (01 such that f(x^k + alpha_kd^k)  f(x^k) + alpha_keta nabla f(x^k)^Td^k\nDefine x^k+1 = x^k + alpha_kx^k\nUpdate k = k + 1 and go to step 2.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using LinearAlgebra\n\nfunction steepest(nlp; itmax=100000, eta=1e-4, eps=1e-6, sigma=0.66)\n  x = nlp.meta.x0\n  fx = obj(nlp, x)\n  ∇fx = grad(nlp, x)\n  slope = dot(∇fx, ∇fx)\n  ∇f_norm = sqrt(slope)\n  iter = 0\n  while ∇f_norm > eps && iter < itmax\n    t = 1.0\n    x_trial = x - t * ∇fx\n    f_trial = obj(nlp, x_trial)\n    while f_trial > fx - eta * t * slope\n      t *= sigma\n      x_trial = x - t * ∇fx\n      f_trial = obj(nlp, x_trial)\n    end\n    x = x_trial\n    fx = f_trial\n    ∇fx = grad(nlp, x)\n    slope = dot(∇fx, ∇fx)\n    ∇f_norm = sqrt(slope)\n    iter += 1\n  end\n  optimal = ∇f_norm <= eps\n  return x, fx, ∇f_norm, optimal, iter\nend\n\nx, fx, ngx, optimal, iter = steepest(nlp)\nprintln(\"x = $x\")\nprintln(\"fx = $fx\")\nprintln(\"ngx = $ngx\")\nprintln(\"optimal = $optimal\")\nprintln(\"iter = $iter\")","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Maybe this code is too complicated? If you're in a class you just want to show a Newton step.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"g(x) = grad(nlp, x)\nH(x) = hess(nlp, x)\nx = nlp.meta.x0\nd = -H(x)\\g(x)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"or a few","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"for i = 1:5\n  global x\n  x = x - H(x)\\g(x)\n  println(\"x = $x\")\nend","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Also, notice how we can reuse the method.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"f(x) = (x[1]^2 + x[2]^2 - 5)^2 + (x[1]*x[2] - 2)^2\nx0 = [3.0; 2.0]\nnlp = ADNLPModel(f, x0)\n\nx, fx, ngx, optimal, iter = steepest(nlp)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"External models can be tested with steepest as well, as long as they implement obj and grad.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"For constrained minimization, you need the constraints vector and bounds too. Bounds on the variables can be passed through a new vector.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using NLPModels, ADNLPModels # hide\nf(x) = (x[1] - 1.0)^2 + 100*(x[2] - x[1]^2)^2\nx0 = [-1.2; 1.0]\nlvar = [-Inf; 0.1]\nuvar = [0.5; 0.5]\nc(x) = [x[1] + x[2] - 2; x[1]^2 + x[2]^2]\nlcon = [0.0; -Inf]\nucon = [Inf; 1.0]\nnlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)\n\nprintln(\"cx = $(cons(nlp, nlp.meta.x0))\")\nprintln(\"Jx = $(jac(nlp, nlp.meta.x0))\")","category":"page"},{"location":"tutorial/#ADNLSModel-tutorial","page":"Tutorial","title":"ADNLSModel tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"In addition to the general nonlinear model, we can define the residual function for a nonlinear least-squares problem. In other words, the objective function of the problem is of the form f(x) = tfrac12F(x)^2, and we can define the function F and its derivatives.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"A simple way to define an NLS problem is with ADNLSModel, which uses automatic differentiation.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using NLPModels, ADNLPModels # hide\nF(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]\nx0 = [-1.2; 1.0]\nnls = ADNLSModel(F, x0, 2) # 2 nonlinear equations","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"residual(nls, x0)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"jac_residual(nls, x0)","category":"page"}]
}
