using LinearAlgebra, Test
using StaticCondensation
S = 2
# Initialize matrices with proper dimensions
A11 = rand(S, S)
A12 = rand(S, S)
A21 = rand(S, S)
A22 = rand(S, S)
A23 = rand(S, S)
A32 = rand(S, S)
A33 = rand(S, S)
A34 = rand(S, S)
A43 = rand(S, S)
A44 = rand(S, S)
Z00 = zeros(S, S)

# Assemble global matrix
A = [A11 A12 Z00 Z00;
     A21 A22 A23 Z00;
     Z00 A32 A33 A34;
     Z00 Z00 A43 A44]

# Initialize right-hand side
a = rand(S)
b = rand(S)
c = rand(S)
d = rand(S)
rhs = [a; b; c; d]
x = A \ rhs

luA44 = lu!(A44)
# Static condensation steps
A33 .-= A34 * (luA44 \ A43)
luS34 = lu!(A33)
A22 .-= A23 * (luS34 \ A32)
luS23 = lu!(A22)
A11 .-= A12 * (luS23 \ A21)
luS12 = lu!(A11)

# Solve system (corrected ordering) to get α with intermediate β, γ, and δ 
δ = luA44 \ d
γ = luS34 \ (c - A34 * δ)
β = luS23 \ (b - A23 * γ)
α = luS12 \ (a - A12 * β)

# Now apply corrections
β -= luS23 \ A21 * α
γ -= luS34 \ A32 * β
δ -= luA44 \ A43 * γ

# Combine solution
y = [α; β; γ; δ]

# Verify solution
@testset "Static Condenstation logic" begin
  @test x[0S+1:1S] ≈ α
  @test x[1S+1:2S] ≈ β
  @test x[2S+1:3S] ≈ γ
  @test x[3S+1:4S] ≈ δ
  @test A * y ≈ rhs # Check residual
end

@testset "Static Condenstation implementation" begin
SC = SCMatrix(A, [0S+1:1S, 1S+1:2S, 2S+1:3S, 3S+1:4S])
SCf = factorise!(SC)
z = zeros(size(x))
ldiv!(z, SCf, rhs)
@test z ≈ x
end
