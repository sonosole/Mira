## probabilistic loss

export CrossEntropy
export CrossEntropyLoss

export BinaryCrossEntropy
export BinaryCrossEntropyLoss

export FocalCE
export FocalCELoss
export FocalBCE
export FocalBCELoss


"""
    CrossEntropy(p::Variable{T}, label::Variable{T}) -> y::Variable{T}
cross entropy is `y = - label * log(p)` where `p` is the output of the network.
"""
function CrossEntropy(p::Variable{T}, label::Variable{T}) where T
    @assert (p.shape == label.shape)
    backprop = (p.backprop || label.backprop)
    𝝆 = ᵛ(label)
    𝒑 = ᵛ(p)
    ϵ = eltype(p)(1e-38)
    y = Variable{T}(- 𝝆 .* log.(𝒑 .+ ϵ), backprop)
    if backprop
        y.backward = function CrossEntropyBackward()
            if need2computeδ!(p)
                δ(p) .-= δ(y) .* 𝝆 ./ (𝒑 .+ ϵ)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    CrossEntropy(p::Variable{T}, label::AbstractArray) -> y::Variable{T}
cross entropy is `y = - label * log(p)` where `p` is the output of the network.
"""
function CrossEntropy(p::Variable{T}, label::AbstractArray) where T
    @assert p.shape == size(label)
    𝝆 = label
    𝒑 = ᵛ(p)
    ϵ = eltype(p)(1e-38)
    y = Variable{T}(- 𝝆 .* log.(𝒑 .+ ϵ), p.backprop)
    if y.backprop
        y.backward = function CrossEntropyBackward()
            if need2computeδ!(p)
                δ(p) .-= δ(y) .* 𝝆 ./ (𝒑 .+ ϵ)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    CrossEntropy(p::AbstractArray, label::AbstractArray) -> lossvalue::AbstractArray
cross entropy is `y = - label * log(p)` where `p` is the output of the network.
"""
function CrossEntropy(p::AbstractArray, label::AbstractArray)
    @assert size(p) == size(label)
    ϵ = eltype(p)(1e-38)
    y = - label .* log.(p .+ ϵ)
    return y
end


"""
    BinaryCrossEntropy(p::Variable{T}, label::Variable{T}) -> y::Variable{T}
binary cross entropy is `y = - label*log(p) - (1-label)*log(1-p)` where `p` is the output of the network.
"""
function BinaryCrossEntropy(p::Variable{T}, label::Variable{T}) where T
    @assert (p.shape == label.shape)
    backprop = (p.backprop || label.backprop)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    𝝆  = ᵛ(label)
    𝒑  = ᵛ(p)
    t₁ = @. -      𝝆  * log(    𝒑 + ϵ)
    t₂ = @. - (𝟙 - 𝝆) * log(𝟙 - 𝒑 + ϵ)
    y  = Variable{T}(t₁ + t₂, backprop)
    if backprop
        y.backward = function binaryCrossEntropyBackward()
            if need2computeδ!(p)
                δ₁ = @. (𝟙 - 𝝆) / (𝟙 - 𝒑 + ϵ)
                δ₂ = @.      𝝆  / (    𝒑 + ϵ)
                δ(p) .+= δ(y) .* (δ₁ - δ₂)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    BinaryCrossEntropy(p::Variable{T}, 𝜌::AbstractArray) -> y::Variable{T}
binary cross entropy is `y = - label*log(p) - (1-label)*log(1-p)` where `p` is the output of the network.
"""
function BinaryCrossEntropy(p::Variable{T}, label::AbstractArray) where T
    @assert p.shape == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    𝝆  = label
    𝒑  = ᵛ(p)
    t₁ = @. -      𝝆  * log(    𝒑 + ϵ)
    t₂ = @. - (𝟙 - 𝝆) * log(𝟙 - 𝒑 + ϵ)
    y  = Variable{T}(t₁ + t₂, p.backprop)
    if y.backprop
        y.backward = function binaryCrossEntropyBackward()
            if need2computeδ!(p)
                δ₁ = @. (𝟙 - 𝝆) / (𝟙 - 𝒑 + ϵ)
                δ₂ = @.      𝝆  / (    𝒑 + ϵ)
                δ(p) .+= δ(y) .* (δ₁ - δ₂)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    BinaryCrossEntropy(p::AbstractArray, label::AbstractArray) -> lossvalue::AbstractArray
binary cross entropy is `y = - label*log(p) - (1-label)*log(1-p)` where `p` is the output of the network.
"""
function BinaryCrossEntropy(p::AbstractArray, label::AbstractArray)
    @assert size(p) == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    t₁ = @. -      label  * log(    p + ϵ)
    t₂ = @. - (𝟙 - label) * log(𝟙 - p + ϵ)
    return t₁ + t₂
end


CrossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss( CrossEntropy(x, label), reduction=reduction )
CrossEntropyLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = loss( CrossEntropy(x, label), reduction=reduction )
CrossEntropyLoss(x::AbstractArray, label::AbstractArray; reduction::String="sum") = loss( CrossEntropy(x, label), reduction=reduction )

BinaryCrossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss(BinaryCrossEntropy(x, label), reduction=reduction)
BinaryCrossEntropyLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = loss(BinaryCrossEntropy(x, label), reduction=reduction)
BinaryCrossEntropyLoss(x::AbstractArray, label::AbstractArray; reduction::String="sum") = loss(BinaryCrossEntropy(x, label), reduction=reduction)


function FocalBCE(p::Variable{T}, label::AbstractArray; gamma::Real=2, alpha::Real=0.5) where T
    @assert p.shape == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    γ  = TO(gamma)
    α  = TO(alpha)
    𝝆  = label
    𝒑  = ᵛ(p)

    w₁ = @. -      α  *      𝝆
    w₂ = @. - (𝟙 - α) * (𝟙 - 𝝆)

    t₁ = @. w₁ * (𝟙 - 𝒑)^ γ * log(    𝒑 + ϵ)
    t₂ = @. w₂ *      𝒑 ^ γ * log(𝟙 - 𝒑 + ϵ)

    y  = Variable{T}(t₁ + t₂, p.backprop)

    if y.backprop
        y.backward = function focalBCEBackward()
            if need2computeδ!(p)
                δ₁ = @. w₁ * (𝟙 - 𝒑)^(γ - 𝟙) * (𝟙 / 𝒑 - γ * log(𝒑) - 𝟙)
                δ₂ = @. w₂ * 𝒑 ^ γ * (𝟙 / (𝒑 - 𝟙) + γ * log(𝟙 - 𝒑) / 𝒑)
                δ(p) .+= δ(y) .* (δ₁ + δ₂)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


function FocalCE(p::Variable{T}, label::AbstractArray; gamma::Real=2) where T
    @assert p.shape == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    γ  = TO(gamma)
    𝝆  = label
    𝒑  = ᵛ(p)

    t = @. - 𝝆 * (𝟙 - 𝒑) ^ γ * log(𝒑 + ϵ)
    y = Variable{T}(t, p.backprop)

    if y.backprop
        y.backward = function focalCEBackward()
            if need2computeδ!(p)
                δ(p) .+= δ(y) .* 𝝆 .* (𝟙 .- 𝒑).^(γ - 𝟙) .* (γ .* log.(𝒑) .+ 𝟙 .- 𝟙 ./ 𝒑)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


FocalCELoss(x::Variable{T}, label::AbstractArray; gamma::Real=2, reduction::String="sum") where T = loss(FocalCE(x, label, gamma=gamma), reduction=reduction)
FocalBCELoss(x::Variable{T}, label::AbstractArray; gamma::Real=2, reduction::String="sum") where T = loss(FocalBCE(x, label, gamma=gamma), reduction=reduction)
