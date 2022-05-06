## probabilistic loss

export crossEntropy
export crossEntropyLoss

export binaryCrossEntropy
export binaryCrossEntropyLoss

export focalCE
export focalCELoss
export focalBCE
export focalBCELoss
export seqfocalCE


"""
    crossEntropy(p::Variable{T}, label::Variable{T}) -> y::Variable{T}
cross entropy is `y = - label * log(p)` where `p` is the output of the network.
"""
function crossEntropy(p::Variable{T}, label::Variable{T}) where T
    @assert (p.shape == label.shape)
    backprop = (p.backprop || label.backprop)
    𝝆 = ᵛ(label)
    𝒑 = ᵛ(p)
    ϵ = eltype(p)(1e-38)
    y = Variable{T}(- 𝝆 .* log.(𝒑 .+ ϵ), backprop)
    if backprop
        y.backward = function crossEntropyBackward()
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
    crossEntropy(p::Variable{T}, label::AbstractArray) -> y::Variable{T}
cross entropy is `y = - label * log(p)` where `p` is the output of the network.
"""
function crossEntropy(p::Variable{T}, label::AbstractArray) where T
    @assert p.shape == size(label)
    𝝆 = label
    𝒑 = ᵛ(p)
    ϵ = eltype(p)(1e-38)
    y = Variable{T}(- 𝝆 .* log.(𝒑 .+ ϵ), p.backprop)
    if y.backprop
        y.backward = function crossEntropyBackward()
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
    crossEntropy(p::AbstractArray, label::AbstractArray) -> lossvalue::AbstractArray
cross entropy is `y = - label * log(p)` where `p` is the output of the network.
"""
function crossEntropy(p::AbstractArray, label::AbstractArray)
    @assert size(p) == size(label)
    ϵ = eltype(p)(1e-38)
    y = - label .* log.(p .+ ϵ)
    return y
end


"""
    binaryCrossEntropy(p::Variable{T}, label::Variable{T}) -> y::Variable{T}
binary cross entropy is `y = - label*log(p) - (1-label)*log(1-p)` where `p` is the output of the network.
"""
function binaryCrossEntropy(p::Variable{T}, label::Variable{T}) where T
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
    binaryCrossEntropy(p::Variable{T}, 𝜌::AbstractArray) -> y::Variable{T}
binary cross entropy is `y = - label*log(p) - (1-label)*log(1-p)` where `p` is the output of the network.
"""
function binaryCrossEntropy(p::Variable{T}, label::AbstractArray) where T
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
    binaryCrossEntropy(p::AbstractArray, label::AbstractArray) -> lossvalue::AbstractArray
binary cross entropy is `y = - label*log(p) - (1-label)*log(1-p)` where `p` is the output of the network.
"""
function binaryCrossEntropy(p::AbstractArray, label::AbstractArray)
    @assert size(p) == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    t₁ = @. -      label  * log(    p + ϵ)
    t₂ = @. - (𝟙 - label) * log(𝟙 - p + ϵ)
    return t₁ + t₂
end


crossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss( crossEntropy(x, label), reduction=reduction )
crossEntropyLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = loss( crossEntropy(x, label), reduction=reduction )
crossEntropyLoss(x::AbstractArray, label::AbstractArray; reduction::String="sum") = loss( crossEntropy(x, label), reduction=reduction )

binaryCrossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = loss(binaryCrossEntropy(x, label), reduction=reduction)
binaryCrossEntropyLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = loss(binaryCrossEntropy(x, label), reduction=reduction)
binaryCrossEntropyLoss(x::AbstractArray, label::AbstractArray; reduction::String="sum") = loss(binaryCrossEntropy(x, label), reduction=reduction)


function focalBCE(p::Variable{T}, label::AbstractArray; gamma::Real=2, alpha::Real=0.5) where T
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


function focalCE(p::Variable{T}, label::AbstractArray; gamma::Real=2) where T
    @assert p.shape == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    γ  = TO(gamma)
    𝝆  = label
    𝒑  = ᵛ(p)

    t = @. 𝝆 * (𝟙 - 𝒑) ^ γ * log(𝒑 + ϵ)
    y = Variable{T}(t, p.backprop)

    if y.backprop
        y.backward = function focalCEBackward()
            if need2computeδ!(p)
                δ(p) .+= δ(y) .* 𝝆 .* (𝟙 .- 𝒑).^(γ - 𝟙) .* (𝟙 ./ 𝒑 .- γ .* log.(𝒑) .- 𝟙)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


focalCELoss(x::Variable{T}, label::AbstractArray; gamma::Real=2, reduction::String="sum") where T = loss(focalCE(x, label, gamma=gamma), reduction=reduction)
focalBCELoss(x::Variable{T}, label::AbstractArray; gamma::Real=2, reduction::String="sum") where T = loss(focalBCE(x, label, gamma=gamma), reduction=reduction)


function seqfocalCE(p::Variable{T},
                    label::AbstractArray,
                    seqlabels::Vector;
                    gamma::Real=2,
                    reduction::String="seqlen") where T

    @assert p.shape == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    γ  = TO(gamma)
    𝝆  = label
    𝒑  = ᵛ(p)

    t = @. 𝝆 * (𝟙 - 𝒑) ^ γ * log(𝒑 + ϵ)
    y = Variable{T}(t, p.backprop)

    if y.backprop
        y.backward = function focalCEBackward()
            if need2computeδ!(p)
                Δ = @. 𝝆 * (𝟙 - 𝒑)^(γ - 𝟙) * (𝟙 / 𝒑 - γ * log(𝒑) - 𝟙)
                reduce3dSeqGrad(Δ, seqlabels, reduction)
                δ(p) .+= δ(y) .* Δ
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end
