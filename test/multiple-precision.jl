function multiple_precision(nlp :: AbstractNLPModel;
                            precisions :: Array = [Float16, Float32, Float64, BigFloat])
  for T in precisions
    x = ones(T, nlp.meta.nvar)
    @test typeof(obj(nlp, x)) == T
    @test eltype(grad(nlp, x)) == T
    @test eltype(hess(nlp, x)) == T
    @test eltype(hess_op(nlp, x)) == T
    rows, cols = hess_structure(nlp)
    vals = hess_coord(nlp, x)
    @test eltype(vals) == T
    Hv = zeros(T, nlp.meta.nvar)
    @test eltype(hess_op!(nlp, rows, cols, vals, Hv)) == T
    if nlp.meta.ncon > 0
      y = ones(T, nlp.meta.ncon)
      @test eltype(cons(nlp, x)) == T
      @test eltype(jac(nlp, x)) == T
      @test eltype(jac_op(nlp, x)) == T
      rows, cols = jac_structure(nlp)
      vals = jac_coord(nlp, x)
      @test eltype(vals) == T
      Av = zeros(T, nlp.meta.ncon)
      Atv = zeros(T, nlp.meta.nvar)
      @test eltype(jac_op!(nlp, rows, cols, vals, Av, Atv)) == T
      @test eltype(hess(nlp, x, y)) == T
      @test eltype(hess(nlp, x, y, obj_weight=one(T))) == T
      @test eltype(hess_op(nlp, x, y)) == T
      rows, cols = hess_structure(nlp)
      vals = hess_coord(nlp, x, y)
      @test eltype(vals) == T
      Hv = zeros(T, nlp.meta.nvar)
      @test eltype(hess_op!(nlp, rows, cols, vals, Hv)) == T
    end
    @info "    $T ✓ "
  end
end