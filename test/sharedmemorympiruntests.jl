using MPI
MPI.Init()
include("common.jl")
using LinearAlgebra, Test, Random
using StaticCondensation
Random.seed!(0)
const S = 400

function sharedmemorympimatrix(tmparray, sharedcomm, sharedrank)
  dimslocal = sharedrank == 0 ? size(tmparray) : (0, 0)
  win, arrayptr = MPI.Win_allocate_shared(Array{Float64}, prod(dimslocal), sharedcomm)
  MPI.Barrier(sharedcomm)
  A = MPI.Win_shared_query(Array{Float64}, prod(size(tmparray)), win; rank=0)
  A = reshape(A, size(tmparray))
  sharedrank == 0 && (A .= tmparray)
  MPI.Barrier(sharedcomm)
  return A, win
end

using StatProfilerHTML
@testset "Static Condenstation implementation" begin
  comm = MPI.COMM_WORLD
  commsize = MPI.Comm_size(comm)
  rank = MPI.Comm_rank(comm)
  A, x, rhs = setupproblem(S)
  A, win = sharedmemorympimatrix(A, comm, rank)
  context = StaticCondensation.MPIContext(
    StaticCondensation.SharedMemoryMPI(win), comm, rank, commsize)
  SC = SCMatrix(A, [0S+1:1S, 1S+1:2S, 2S+1:3S, 3S+1:4S]; context=context)
  SCf = factorise!(SC)
  z = zeros(size(x))
  ldiv!(z, SCf, rhs)
  @test z ≈ x
  ldiv!(SCf, rhs)
  @test rhs ≈ x
  @time ldiv!(z, SCf, rhs)
  StaticCondensation.free(SCf)
end

MPI.Finalize()
