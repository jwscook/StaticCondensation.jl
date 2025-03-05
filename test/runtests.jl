using LinearAlgebra, Test

include("common.jl")
using StaticCondensation

const S = 1000
A, x, rhs = setupproblem(S)

using StatProfilerHTML
@testset "Static Condenstation implementation" begin
SC = SCMatrix(A, [0S+1:1S, 1S+1:2S, 2S+1:3S, 3S+1:4S])
SCf = factorise!(SC)
z = zeros(size(x))
ldiv!(z, SCf, rhs)
@test z ≈ x
ldiv!(SCf, rhs)
@test rhs ≈ x

@time ldiv!(z, SCf, rhs)
end
