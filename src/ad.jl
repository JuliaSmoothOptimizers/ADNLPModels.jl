abstract type ADBackend end
struct ForwardDiffAD <: ADBackend end
struct ZygoteAD <: ADBackend end
struct ReverseDiffAD <: ADBackend end

throw_error(b) = throw(ArgumentError("The AD backend $b is not loaded. Please load the corresponding AD package."))
gradient(b::ADBackend, ::Any, ::Any) = throw_error(b)
gradient!(b::ADBackend, ::Any, ::Any, ::Any) = throw_error(b)
jacobian(b::ADBackend, ::Any, ::Any) = throw_error(b)
hessian(b::ADBackend, ::Any, ::Any) = throw_error(b)
directional_derivative(b::ADBackend, ::Any, ::Any, ::Any) = throw_error(b)
pullback(b::ADBackend, ::Any, ::Any, ::Any) = throw_error(b)
function directional_second_derivative(::ADBackend, f, x, v, w)
    return ForwardDiff.derivative(
        t -> ForwardDiff.derivative(
            s -> f(x + s * w + t * v), 0,
        ), 0,
    )
end
function Hvprod(b::ADBackend, f, x, v)
    return ForwardDiff.derivative(t -> gradient(b, f, x + t * v), 0)
end

gradient(::ForwardDiffAD, f, x) = ForwardDiff.gradient(f, x)
function gradient!(::ForwardDiffAD, g, f, x)
    return ForwardDiff.gradient!(g, f, x)
end
jacobian(::ForwardDiffAD, f, x) = ForwardDiff.jacobian(f, x)
hessian(::ForwardDiffAD, f, x) = ForwardDiff.hessian(f, x)
function directional_derivative(::ForwardDiffAD, f, x, v)
    return ForwardDiff.derivative(t -> f(x + t * v), 0)
end
function pullback(::ForwardDiffAD, f, x, v)
    return ForwardDiff.gradient(x -> dot(f(x), v), x)
end

@init begin
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        function gradient(::ZygoteAD, f, x)
            g = Zygote.gradient(f, x)[1]
            return g === nothing ? zero(x) : g
        end
        function gradient!(::ZygoteAD, g, f, x)
            _g = Zygote.gradient(f, x)[1]
            g .= _g === nothing ? 0 : _g
        end
        function jacobian(::ZygoteAD, f, x)
            return Zygote.jacobian(f, x)[1]
        end
        function hessian(b::ZygoteAD, f, x)
            return jacobian(ForwardDiffAD(), x -> gradient(b, f, x), x)
        end
        function directional_derivative(::ZygoteAD, f, x, v)
            return Zygote.jacobian(t -> f(x + t * v), 0)[1]
        end
        function pullback(::ZygoteAD, f, x, v)
            g = Zygote.gradient(x -> dot(f(x), v), x)[1]
            return g === nothing ? zero(x) : g
        end
    end
    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        gradient(::ReverseDiffAD, f, x) = ReverseDiff.gradient(f, x)
        function gradient!(::ReverseDiffAD, g, f, x)
            return ReverseDiff.gradient!(g, f, x)
        end
        jacobian(::ReverseDiffAD, f, x) = ReverseDiff.jacobian(f, x)
        hessian(::ReverseDiffAD, f, x) = ReverseDiff.hessian(f, x)
        function directional_derivative(::ReverseDiffAD, f, x, v)
            return ReverseDiff.derivative(t -> f(x + t * v), 0)
        end
        function pullback(::ReverseDiffAD, f, x, v)
            return ReverseDiff.gradient(x -> dot(f(x), v), x)
        end
    end
end
