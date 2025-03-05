using MPI
MPI.Init()
include("common.jl")
using LinearAlgebra, Test, Random
using StaticCondensation
Random.seed!(0)

const S = 400
A, x, rhs = setupproblem(S)

using StatProfilerHTML
@testset "Static Condenstation implementation" begin
  comm = MPI.COMM_WORLD
  commsize = MPI.Comm_size(comm)
  rank = MPI.Comm_rank(comm)
  context = StaticCondensation.MPIContext(
    StaticCondensation.DistributedMemoryMPI(), comm, rank, commsize)
  SC = SCMatrix(A, [0S+1:1S, 1S+1:2S, 2S+1:3S, 3S+1:4S]; context=context)
  SCf = factorise!(SC)
  z = zeros(size(x))
  ldiv!(z, SCf, rhs)
  @test z ≈ x
  ldiv!(SCf, rhs)
  @test rhs ≈ x
  @time ldiv!(z, SCf, rhs)
end

MPI.Finalize()
