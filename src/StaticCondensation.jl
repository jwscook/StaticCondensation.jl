module StaticCondensation

using LinearAlgebra, Test

export SCMatrix, SCMatrixFactorisation, factorise!

struct SCMatrix{T, M<:AbstractMatrix{T}} <: AbstractMatrix{T}
  A::M
  blocks::Vector{UnitRange{Int64}}
  work::M
  function SCMatrix(A::M, blocks) where M<:AbstractMatrix
    work = similar(A, maximum(length.(blocks)), maximum(length.(blocks)))
    return new{eltype(M), M}(A, blocks, work)
  end
end
Base.getindex(A::SCMatrix, i, j) = A.A[i, j]
Base.setindex!(A::SCMatrix, v, i, j) = A.A[i, j] = v
 
struct SCMatrixFactorisation{T, M<:AbstractMatrix{T},U} <: AbstractMatrix{T}
  A::SCMatrix{T,M}
  lus::U
end
Base.size(SC::SCMatrix) = size(SC.A)
Base.size(SC::SCMatrix, i) = size(SC.A, i)

function factorise!(A::SCMatrix{T}) where T
  i = A.blocks[end]
  lun = lu!(view(A.A, i, i))
  lus = Vector{typeof(lun)}(undef, length(A.blocks))
  lus[end] = lun
  for (ci, i) in enumerate(reverse(A.blocks))
    ci == 1 && continue
    c = length(A.blocks) - ci + 1
    i1 = A.blocks[c+1]
    ldiv!(view(A.work, 1:length(i), 1:length(i)), lus[c+1], A.A[i1, i])
    @views mul!(A.A[i, i],
                A.A[i, i1],
                view(A.work, 1:length(i), 1:length(i)),
                -one(T),
                one(T))
    lus[c] = lu!(view(A.A, i, i))
  end
  return SCMatrixFactorisation(A, lus)
end

LinearAlgebra.ldiv!(A::SCMatrix{T,M}, b::AbstractArray) where {T,M} = ldiv!(fatorise!(A), b)
LinearAlgebra.ldiv!(x::AbstractArray, A::SCMatrix{T,M}, b::AbstractArray) where {T,M} = ldiv!(x, factorise!(A), b)
LinearAlgebra.ldiv!(F::SCMatrixFactorisation{T,M}, b::AbstractArray) where {T,M} = ldiv!(b, F, deepcopy(b))
function LinearAlgebra.ldiv!(x::AbstractArray, F::SCMatrixFactorisation{T,M}, b::AbstractArray) where {T,M}
  i = F.A.blocks[end]
  view(x, i, :) .= F.lus[end] \ view(b, i, :)
  @views for (ci, i) in enumerate(reverse(F.A.blocks))
    ci == 1 && continue
    xi = view(x, i, :)
    bi = view(b, i, :)
    c = length(F.A.blocks) - ci + 1
    mul!(xi, F.A.A[i, F.A.blocks[c+1]], x[F.A.blocks[c+1], :])
    xi .= bi .- xi
    ldiv!(xi, F.lus[c], xi)
  end
  @views for (c, i) in enumerate(F.A.blocks)
    c == 1 && continue
    xi = view(x, i, :)
    tmp = view(F.A.work, 1:length(i), 1:size(x, 2))
    mul!(tmp, F.A.A[i, F.A.blocks[c-1]], x[F.A.blocks[c-1], :])
    ldiv!(F.lus[c], tmp)
    BLAS.axpy!(-one(T), tmp, xi)
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
