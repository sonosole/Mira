## probabilistic loss

export crossEntropy
export crossEntropyLoss

export binaryCrossEntropy
export binaryCrossEntropyLoss


"""
    crossEntropy(p::Variable{T}, 𝜌::Variable{T}) -> y::Variable{T}
cross entropy is `y = - 𝜌 * log(p) where 𝜌 is the target and p is the output of the network.
"""
function crossEntropy(p::Variable{T}, 𝜌::Variable{T}) where T
    @assert (p.shape == 𝜌.shape)
    backprop = (p.backprop || 𝜌.backprop)
    ϵ = eltype(p)(1e-38)
    y = Variable{T}(- ᵛ(𝜌) .* log.(ᵛ(p) .+ ϵ), backprop)
    if backprop
        y.backward = function crossEntropyBackward()
            if need2computeδ!(p)
                δ(p) .-= δ(y) .* ᵛ(𝜌) ./ (ᵛ(p) .+ ϵ)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    crossEntropy(p::Variable{T}, 𝜌::AbstractArray) -> y::Variable{T}
cross entropy is `y = - 𝜌 * log(p) where 𝜌 is the target and p is the output of the network.
"""
function crossEntropy(p::Variable{T}, 𝜌::AbstractArray) where T
    @assert p.shape == size(𝜌)
    ϵ = eltype(p)(1e-38)
    y = Variable{T}(- 𝜌 .* log.(ᵛ(p) .+ ϵ), p.backprop)
    if y.backprop
        y.backward = function crossEntropyBackward()
            if need2computeδ!(p)
                δ(p) .-= δ(y) .* 𝜌 ./ (ᵛ(p) .+ ϵ)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    binaryCrossEntropy(p::Variable{T}, 𝜌::Variable{T}) -> y::Variable{T}
binary cross entropy is `y = - 𝜌log(p) - (1-𝜌)log(1-p)` where 𝜌 is the target and p is the output of the network.
"""
function binaryCrossEntropy(p::Variable{T}, 𝜌::Variable{T}) where T
    @assert (p.shape == 𝜌.shape)
    backprop = (p.backprop || 𝜌.backprop)
    TOO  = eltype(p)
    ϵ  = TOO(1e-38)
    𝟙  = TOO(1.0f0)
    t₁ = -       ᵛ(𝜌)  .* log.(     ᵛ(p) .+ ϵ)
    t₂ = - (𝟙 .- ᵛ(𝜌)) .* log.(𝟙 .- ᵛ(p) .+ ϵ)
    y  = Variable{T}(t₁ + t₂, backprop)
    if backprop
        y.backward = function binaryCrossEntropyBackward()
            if need2computeδ!(p)
                δ₁ = (𝟙 .- ᵛ(𝜌)) ./ (𝟙 .- ᵛ(p) .+ ϵ)
                δ₂ =       ᵛ(𝜌)  ./ (     ᵛ(p) .+ ϵ)
                δ(p) .+= δ(y) .* (δ₁ - δ₂)
            end
            ifNotKeepδThenFreeδ!(y);
        end
        addchild(y, p)
    end
    return y
end


"""
    binaryCrossEntropy(p::Variable{T}, 𝜌::AbstractArray) -> y::Variable{T}
binary cross entropy is `y = - 𝜌log(p) - (1-𝜌)log(1-p)` where 𝜌 is the target and p is the output of the network.
"""
function binaryCrossEntropy(p::Variable{T}, 𝜌::AbstractArray) where T
    @assert p.shape == size(𝜌)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    t₁ = -       𝜌  .* log.(     ᵛ(p) .+ ϵ)
    t₂ = - (𝟙 .- 𝜌) .* log.(𝟙 .- ᵛ(p) .+ ϵ)
    y  = Variable{T}(t₁ + t₂, p.backprop)
    if y.backprop
        y.backward = function binaryCrossEntropyBackward()
            if need2computeδ!(p)
                δ₁ = (𝟙 .- 𝜌) ./ (𝟙 .- ᵛ(p) .+ ϵ)
                δ₂ =       𝜌  ./ (     ᵛ(p) .+ ϵ)
                δ(p) .+= δ(y) .* (δ₁ - δ₂)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


crossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss( crossEntropy(x, label), reduction=reduction )
crossEntropyLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = loss( crossEntropy(x, label), reduction=reduction )

binaryCrossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss(binaryCrossEntropy(x, label), reduction=reduction)
binaryCrossEntropyLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = loss(binaryCrossEntropy(x, label), reduction=reduction)
