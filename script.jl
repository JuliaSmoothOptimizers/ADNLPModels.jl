using ADNLPModels
F! = (Fx, x) -> begin Fx[1] = x[1]; Fx[2] = x[2] end
newF! = (Fx, x) -> begin Fx[1] = x[1]; Fx[2] = x[2] end
nls = ADNLSModel!(F!, ones(2), 2)
back = nls.adbackend
new_back = ADNLPModels.ADModelNLSBackend(2, newF!, 2,
    gradient_backend = ADNLPModels.EmptyADbackend(),
    hprod_backend = ADNLPModels.EmptyADbackend(),
    hessian_backend = ADNLPModels.EmptyADbackend(),
    hprod_residual_backend = ADNLPModels.EmptyADbackend(),
    jprod_residual_backend = ADNLPModels.ForwardDiffADJprod, # or whatever you use
    jtprod_residual_backend = ADNLPModels.ForwardDiffADJtprod, # or whatever you use
    jacobian_residual_backend = ADNLPModels.SparseADJacobian, # or whatever you use
    hessian_residual_backend = ADNLPModels.EmptyADbackend(),
)
set_adbackend!(nls, new_back)
