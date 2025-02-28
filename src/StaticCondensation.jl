module StaticCondensation

using LinearAlgebra, Test

export SCMatrix, SCMatrixFactorisation, factorise!

struct SCMatrix{T, M<:AbstractMatrix{T}} <: AbstractMatrix{T}
  A::M
  blocks::Vector{UnitRange{Int64}}
end
Base.getindex(A::SCMatrix, i, j) = A.A[i, j]
Base.setindex!(A::SCMatrix, v, i, j) = A.A[i, j] = v
 
struct SCMatrixFactorisation{T, M<:AbstractMatrix{T},U} <: AbstractMatrix{T}
  A::SCMatrix{T,M}
  lus::U
end
Base.size(SC::SCMatrix) = size(SC.A)
Base.size(SC::SCMatrix, i) = size(SC.A, i)

function factorise!(A::SCMatrix)
  i = A.blocks[end]
  lun = lu!(view(A.A, i, i))
  lus = Vector{typeof(lun)}(undef, length(A.blocks))
  lus[end] = lun
  for (ci, i) in enumerate(reverse(A.blocks))
    ci == 1 && continue
    c = length(A.blocks) - ci + 1
    i1 = A.blocks[c+1]
    @views A.A[i, i] .-= A.A[i, i1] * (lus[c+1] \ A.A[i1, i])
    lus[c] = lu!(view(A.A, i, i))
  end
  return SCMatrixFactorisation(A, lus)
end

function LinearAlgebra.ldiv!(x::AbstractArray, F::SCMatrixFactorisation{T,M}, b::AbstractArray) where {T,M}
  i = F.A.blocks[end]
  view(x, i, :) .= F.lus[end] \ view(b, i, :)
  @views for (ci, i) in enumerate(reverse(F.A.blocks))
    ci == 1 && continue
    c = length(F.A.blocks) - ci + 1
    x[i, :] .= F.lus[c] \ (b[i, :]  .- F.A.A[i, F.A.blocks[c+1]] * x[F.A.blocks[c+1], :])
  end
  @views for (c, i) in enumerate(F.A.blocks)
    c == 1 && continue
    x[i, :] .-= F.lus[c] \ F.A.A[i, F.A.blocks[c-1]] * x[F.A.blocks[c-1], :]
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
