function Base.floor(x::Variable{T}; digits::Int=0) where T
    y = Variable{T}(floor.(ᵛ(x); digits), x.backprop)
    if y.backprop
        y.backward = function ∇floor()
            if need2computeδ!(x)
                x ← δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.ceil(x::Variable{T}; digits::Int=0) where T
    y = Variable{T}(ceil.(ᵛ(x); digits), x.backprop)
    if y.backprop
        y.backward = function ∇ceil()
            if need2computeδ!(x)
                x ← δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.round(x::Variable{T}; digits::Int=0) where T
    y = Variable{T}(round.(ᵛ(x); digits), x.backprop)
    if y.backprop
        y.backward = function ∇round()
            if need2computeδ!(x)
                x ← δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.trunc(x::Variable{T}; digits::Int=0) where T
    y = Variable{T}(trunc.(ᵛ(x); digits), x.backprop)
    if y.backprop
        y.backward = function ∇trunc()
            if need2computeδ!(x)
                x ← δ(y)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


Base.floor(x::AbstractArray; digits::Int=0) = floor.(x;digits)
Base.ceil(x::AbstractArray;  digits::Int=0) = ceil.(x;digits)
Base.round(x::AbstractArray; digits::Int=0) = round.(x;digits)
Base.trunc(x::AbstractArray; digits::Int=0) = trunc.(x;digits)
