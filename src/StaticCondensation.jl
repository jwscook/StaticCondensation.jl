module StaticCondensation

using LinearAlgebra, Test

export SCMatrix, SCMatrixFactorisation, factorise!

using MPI, ChunkSplitters

abstract type AbstractContext end
struct SerialContext <: AbstractContext end
(s::SerialContext)() = nothing

abstract type AbstractMPI end
struct DistributedMemoryMPI <: AbstractMPI end
struct SharedMemoryMPI{T} <: AbstractMPI
  win::T
end

struct MPIContext{T<:AbstractMPI, C} <: AbstractContext
  mpitype::T
  comm::C
  rank::Int
  size::Int
end
(con::MPIContext)() = MPI.Barrier(con.comm)

free(con::MPIContext{<:DistributedMemoryMPI}) = MPI.free(con.mpitype.win)
free(x) = nothing

struct SCMatrix{T, M<:AbstractMatrix{T}, C<:AbstractContext} <: AbstractMatrix{T}
  A::M
  blocks::Vector{UnitRange{Int64}}
  work::M
  context::C
  function SCMatrix(A::M, blocks; context=SerialContext()) where M<:AbstractMatrix
    work = similar(A, maximum(length.(blocks)), maximum(length.(blocks)))
    fill!(work, 0)
    return new{eltype(A), M, typeof(context)}(A, blocks, work, context)
  end
end
Base.getindex(A::SCMatrix, i, j) = A.A[i, j]
Base.setindex!(A::SCMatrix, v, i, j) = A.A[i, j] = v

free(A::SCMatrix) = free(A.context)
 
struct SCMatrixFactorisation{T, M<:AbstractMatrix{T},U} <: AbstractMatrix{T}
  A::SCMatrix{T,M}
  lus::U
end
Base.size(SC::SCMatrix) = size(SC.A)
Base.size(SC::SCMatrix, i) = size(SC.A, i)

free(A::SCMatrixFactorisation) = free(A.A)

mulwrapper!(C, A, B, α, β, ::SerialContext) = mul!(C, A, B, α, β)
mulwrapper!(C, A, B, ::SerialContext) = mul!(C, A, B)
function mpireduce!(C, allrows, context::MPIContext)
  for src in 0:context.size - 1
    rows = allrows[src + 1]
    C[rows, :] .= MPI.bcast(C[rows, :], src, context.comm)
  end
## gave up trying this method
#  counts = MPI.Allgather(Int32(length(view(C, allrows[context.rank + 1], :)), context.comm)
#  MPI.Allgatherv!(MPI.IN_PLACE, C, counts, context.comm)
end
finalisemulwrapper!(C, rows, context::MPIContext{<:SharedMemoryMPI}) = context() # just a barrier
finalisemulwrapper!(C, rows, context::MPIContext{<:DistributedMemoryMPI}) = mpireduce!(C, rows, context)

function mulwrapper!(C, A, B, α, β, context::MPIContext)
  allrows = chunks(1:size(C, 1), n=context.size)
  rows = allrows[context.rank + 1]
  @views mul!(C[rows, :], A[rows, :], B, α, β)
  finalisemulwrapper!(C, allrows, context)
  return C
end
function mulwrapper!(C, A, B, context::MPIContext)
  allrows = chunks(1:size(C, 1), n=context.size)
  rows = allrows[context.rank + 1]
  @views mul!(C[rows, :], A[rows, :], B)
  finalisemulwrapper!(C, allrows, context)
  return C
end
reductionnonsharedarray!(x, context::SerialContext) = nothing
reductionnonsharedarray!(x, context::MPIContext{<:DistributedMemoryMPI}) = nothing
function reductionnonsharedarray!(x, context::MPIContext{<:SharedMemoryMPI})
  allrows = chunks(1:size(x, 1), n=context.size)
  mpireduce!(x, allrows, context)
end

luwrapper!(A, context) = (luA = lu!(A); context(); return luA)
function luwrapper!(A, context::MPIContext{<:SharedMemoryMPI})
  luA = context.rank == 0 ? lu!(A) : nothing
  return MPI.bcast(luA, 0, context.comm)
end

function factorise!(A::SCMatrix{T}) where T
  i = A.blocks[end]
  lun = luwrapper!(view(A.A, i, i), A.context)
  lus = Vector{typeof(lun)}(undef, length(A.blocks))
  lus[end] = lun
  for (ci, i) in enumerate(reverse(A.blocks))
    ci == 1 && continue
    c = length(A.blocks) - ci + 1
    i1 = A.blocks[c+1]
    ldiv!(view(A.work, 1:length(i), 1:length(i)), lus[c+1], A.A[i1, i])
    @views mulwrapper!(A.A[i, i], A.A[i, i1], view(A.work, 1:length(i), 1:length(i)), -one(T), one(T), A.context)
    lus[c] = luwrapper!(view(A.A, i, i), A.context)
  end
  return SCMatrixFactorisation(A, lus)
end

LinearAlgebra.ldiv!(A::SCMatrix{T,M}, b::AbstractArray) where {T,M} = ldiv!(factorise!(A), b)
LinearAlgebra.ldiv!(x::AbstractArray, A::SCMatrix{T,M}, b::AbstractArray) where {T,M} = ldiv!(x, factorise!(A), b)
LinearAlgebra.ldiv!(F::SCMatrixFactorisation{T,M}, b::AbstractArray) where {T,M} = ldiv!(b, F, deepcopy(b))
function LinearAlgebra.ldiv!(x::AbstractArray, F::SCMatrixFactorisation{T,M}, b::AbstractArray) where {T,M}
  i = F.A.blocks[end]
  view(x, i, :) .= F.lus[end] \ view(b, i, :)
  @views for (ci, i) in enumerate(reverse(F.A.blocks)) # serial loop
    ci == 1 && continue
    xi = view(x, i, :)
    bi = view(b, i, :)
    c = length(F.A.blocks) - ci + 1
    mulwrapper!(xi, F.A.A[i, F.A.blocks[c+1]], x[F.A.blocks[c+1], :], F.A.context)
    reductionnonsharedarray!(xi, F.A.context) # tmp is never a sharedmemory mpi array so must be reduced
    xi .= bi .- xi
    ldiv!(xi, F.lus[c], xi)
  end
  @views for (c, i) in enumerate(F.A.blocks) # serial loop
    c == 1 && continue
    xi = view(x, i, :)
    tmp = view(F.A.work, 1:length(i), 1:size(x, 2))
    mulwrapper!(tmp, F.A.A[i, F.A.blocks[c-1]], x[F.A.blocks[c-1], :], F.A.context)
    reductionnonsharedarray!(tmp, F.A.context) # tmp is never a sharedmemory mpi array so must be reduced
    ldiv!(F.lus[c], tmp)
    xi .-= tmp
    #BLAS.axpy!(-one(T), tmp, xi) # errors out annoying due to view related dispatch
  end
  return x
end
function LinearAlgebra.:\(A::SCMatrixFactorisation{T,M}, b::AbstractVecOrMat
    ) where {T,M}
  x = similar(b)
  ldiv!(x, A, b)
  return x
end

end
