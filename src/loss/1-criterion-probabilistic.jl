## probabilistic loss

export crossEntropy
export crossEntropyLoss

export binaryCrossEntropy
export binaryCrossEntropyLoss

export focalBCE
export focalBCELoss


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


"""
    binaryCrossEntropy(p::AbstractArray, label::AbstractArray) -> lossvalue::AbstractArray
binary cross entropy is `y = - label*log(p) - (1-label)*log(1-p)` where p is the output of the network.
"""
function binaryCrossEntropy(p::AbstractArray, label::AbstractArray)
    @assert size(p) == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    t₁ = -       label  .* log.(     p .+ ϵ)
    t₂ = - (𝟙 .- label) .* log.(𝟙 .- p .+ ϵ)
    return t₁ + t₂
end


"""
    crossEntropyLoss(p::AbstractArray, label::AbstractArray) -> lossvalue::AbstractArray
cross entropy is `y = - label * log(p) where p is the output of the network.
"""
function crossEntropyLoss(p::AbstractArray, label::AbstractArray)
    @assert size(p) == size(label)
    ϵ = eltype(p)(1e-38)
    y = - label .* log.(p .+ ϵ)
    return y
end


crossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss( crossEntropy(x, label), reduction=reduction )
crossEntropyLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = loss( crossEntropy(x, label), reduction=reduction )
crossEntropyLoss(x::AbstractArray, label::AbstractArray; reduction::String="sum") = loss( crossEntropy(x, label), reduction=reduction )

binaryCrossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss(binaryCrossEntropy(x, label), reduction=reduction)
binaryCrossEntropyLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = loss(binaryCrossEntropy(x, label), reduction=reduction)
binaryCrossEntropyLoss(x::AbstractArray, label::AbstractArray; reduction::String="sum") = loss(binaryCrossEntropy(x, label), reduction=reduction)


function focalBCE(p::Variable{T}, 𝜌::AbstractArray; gamma::Real=2, alpha::Real=0.5) where T
    @assert p.shape == size(𝜌)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    γ  = gamma
    α  = alpha
    𝒑  = ᵛ(p)

    w₁ = @. -      α  *      𝜌
    w₂ = @. - (𝟙 - α) * (𝟙 - 𝜌)

    t₁ = @. w₁ * (𝟙 - 𝒑)^ γ * log(    𝒑 + ϵ)
    t₂ = @. w₂ *      𝒑 ^ γ * log(𝟙 - 𝒑 + ϵ)

    y  = Variable{T}(t₁ + t₂, p.backprop)

    if y.backprop
        y.backward = function focalBCEBackward()
            if need2computeδ!(p)
                δ₁ = @. w₁ * (𝟙 - 𝒑)^(γ-1) * (𝟙 / 𝒑 - γ * log(𝒑) - 𝟙)
                δ₂ = @. w₂ * 𝒑 ^ γ * (𝟙 / (𝒑 - 𝟙) + γ * log(𝟙 - 𝒑) / 𝒑)
                δ(p) .+= δ(y) .* (δ₁ + δ₂)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


focalBCELoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = loss(focalBCE(x, label), reduction=reduction)
