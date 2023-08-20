function invpermdims(pdims::Vector{Int})
    newdims = Vector{Int}(undef, length(pdims))
    for (i, d) in enumerate(pdims)
        @inbounds newdims[d] = i
    end
    return newdims
end

function invpermdims(pdims::NTuple{D,Int}) where D
    newdims = Vector{Int}(undef, D)
    for (i, d) in enumerate(pdims)
        @inbounds newdims[d] = i
    end
    return newdims
end

function Base.permutedims(x::Variable{T}, d::Vector{Int}) where T
    assertdim(x, length(d))
    y = Variable{T}(permutedims(ᵛ(x), d), x.backprop)

    if y.backprop
        y.backward = function ∇permutedims()
            if need2computeδ!(x)
                x ← permutedims(δ(y), invpermdims(d))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.permutedims(x::Variable{T}, d::NTuple{D, Int}) where {T,D}
    assertdim(x, D)
    y = Variable{T}(permutedims(ᵛ(x), d), x.backprop)

    if y.backprop
        y.backward = function ∇permutedims()
            if need2computeδ!(x)
                x ← permutedims(δ(y), invpermdims(d))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.adjoint(x::Variable{T}) where T
    y = Variable{T}(ᵛ(x)', x.backprop)
    if y.backprop
        y.backward = function ∇adjoint()
            if need2computeδ!(x)
                x ← δ(y)'
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.transpose(x::Variable{T}) where T
    y = Variable{T}(transpose(ᵛ(x)), x.backprop)
    if y.backprop
        y.backward = function ∇transpose()
            if need2computeδ!(x)
                x ← transpose(ᵟ(y))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.reshape(x::Variable{T}, newsize::Dims{N}) where {T,N}
    y = Variable{T}( reshape(ᵛ(x), newsize), x.backprop )
    if y.backprop
        y.backward = function ∇reshape()
            if need2computeδ!(x)
                x ← reshape(δ(y), x.shape)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.reshape(x::Variable{T}, shape::Int...) where T
    y = Variable{T}( reshape(ᵛ(x), shape), x.backprop )
    if y.backprop
        y.backward = function ∇reshape()
            if need2computeδ!(x)
                x ← reshape(δ(y), x.shape)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.reshape(x::Variable{T}, s₁::Int, s₂::Int) where T
    y = Variable{T}( reshape(ᵛ(x), s₁, s₂), x.backprop )
    if y.backprop
        y.backward = function ∇reshape()
            if need2computeδ!(x)
                x ← reshape(δ(y), x.shape)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

function Base.reshape(x::Variable{T}, s₁::Int, s₂::Int, s₃::Int) where T
    y = Variable{T}( reshape(ᵛ(x), s₁, s₂, s₃), x.backprop )
    if y.backprop
        y.backward = function ∇reshape()
            if need2computeδ!(x)
                x ← reshape(δ(y), x.shape)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end
