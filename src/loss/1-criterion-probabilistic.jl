## probabilistic loss

export crossEntropy
export crossEntropyLoss

export binaryCrossEntropy
export binaryCrossEntropyLoss


"""
    crossEntropy(x::Variable{T}, label::Variable{T}) -> Variable{T}
cross entropy = - y * log(̂y) where y is target and ̂y is the output of the network.
"""
function crossEntropy(x::Variable{T}, label::Variable{T}) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    ϵ = eltype(x)(1e-38)
    y = Variable{T}(- ᵛ(label) .* log.(ᵛ(x) .+ ϵ), backprop)
    if backprop
        y.backward = function crossEntropyBackward()
            if need2computeδ!(x)
                δ(x) .-= δ(y) .* ᵛ(label) ./ (ᵛ(x) .+ ϵ)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

"""
    crossEntropy(x::Variable{T}, label::T) -> Variable{T}
cross entropy = - y * log(̂y) where y is target and ̂y is the output of the network.
"""
function crossEntropy(x::Variable{T}, label::T) where T
    @assert x.shape == size(label)
    ϵ = eltype(x)(1e-38)
    y = Variable{T}(- label .* log.(ᵛ(x) .+ ϵ), x.backprop)
    if y.backprop
        y.backward = function crossEntropyBackward()
            if need2computeδ!(x)
                δ(x) .-= δ(y) .* label ./ (ᵛ(x) .+ ϵ)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


crossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss( crossEntropy(x, label), reduction=reduction )
crossEntropyLoss(x::Variable{T}, label::T; reduction::String="sum") where T = loss( crossEntropy(x, label), reduction=reduction )


"""
    binaryCrossEntropy(x::Variable{T}, l::Variable{T}) -> Variable{T}
binary cross entropy = - y * log(̂y) - (1 - y) * log(1-̂y)
"""
function binaryCrossEntropy(x::Variable{T}, label::Variable{T}) where T
    @assert (x.shape == label.shape)
    backprop = (x.backprop || label.backprop)
    TOO  = eltype(x)
    ϵ  = TOO(1e-38)
    𝟙  = TOO(1.0f0)
    tmp1 = - ᵛ(label) .* log.(ᵛ(x) .+ ϵ)
    tmp2 = - (𝟙 .- ᵛ(label)) .* log.(𝟙 .- ᵛ(x) .+ ϵ)
    y  = Variable{T}(tmp1 + tmp2, backprop)
    if backprop
        y.backward = function binaryCrossEntropyBackward()
            if need2computeδ!(x)
                temp1 = (𝟙 .- ᵛ(label)) ./ (𝟙 .- ᵛ(x) .+ ϵ)
                temp2 = ᵛ(label) ./ (ᵛ(x) .+ ϵ)
                δ(x) .+= δ(y) .* (temp1 - temp2)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        addchild(y, x)
    end
    return y
end


"""
    binaryCrossEntropy(x::Variable{T}, l::T) -> Variable{T}
binary cross entropy = - y * log(̂y) - (1 - y) * log(1-̂y)
"""
function binaryCrossEntropy(x::Variable{T}, label::T) where T
    @assert x.shape == size(label.shape)
    TO = eltype(x)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    tmp1 = - label .* log.(ᵛ(x) .+ ϵ)
    tmp2 = - (𝟙 .- label) .* log.(𝟙 .- ᵛ(x) .+ ϵ)
    y  = Variable{T}(tmp1 + tmp2, x.backprop)
    if y.backprop
        y.backward = function binaryCrossEntropyBackward()
            if need2computeδ!(x)
                temp1 = (𝟙 .- label) ./ (𝟙 .- ᵛ(x) .+ ϵ)
                temp2 = label ./ (ᵛ(x) .+ ϵ)
                δ(x) .+= δ(y) .* (temp1 - temp2)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


binaryCrossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss( binaryCrossEntropy(x, label), reduction=reduction)
binaryCrossEntropyLoss(x::Variable{T}, label::T; reduction::String="sum") where T = loss( binaryCrossEntropy(x, label), reduction=reduction)
