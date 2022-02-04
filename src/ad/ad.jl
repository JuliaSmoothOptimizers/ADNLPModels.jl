export ADBackend

abstract type ADBackend end

include("forward.jl")
include("reverse.jl")
