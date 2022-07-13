## probabilistic loss

export CrossEntropy
export CrossEntropyLoss

export BinaryCrossEntropy
export BinaryCrossEntropyLoss

export FocalCE
export FocalCELoss
export FocalBCE
export FocalBCELoss

export NLogCrossEntropy
export NLogCELoss

export InvPowerCrossEntropy
export InvPowerCELoss


"""
    CrossEntropy(p::Variable{T}, label::Variable{T}) -> y::Variable{T}
cross entropy is `y = - label * log(p)` where `p` is the output of the network.
"""
function CrossEntropy(p::Variable{T}, label::Variable{T}) where T
    @assert (p.shape == label.shape)
    backprop = (p.backprop || label.backprop)
    𝝆 = ᵛ(label)
    𝒑 = ᵛ(p) .+ eltype(p)(1e-38)
    y = Variable{T}(- 𝝆 .* log.(𝒑), backprop)
    if backprop
        y.backward = function ∇CrossEntropy()
            if need2computeδ!(p)
                δ(p) .-= δ(y) .* 𝝆 ./ 𝒑
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    CrossEntropy(p::Variable, label::AbstractArray) -> y::Variable
cross entropy is `y = - label * log(p)` where `p` is the output of the network.
"""
function CrossEntropy(p::Variable{T}, label::AbstractArray) where T
    @assert p.shape == size(label)
    𝝆 = label
    𝒑 = ᵛ(p) .+ eltype(p)(1e-38)
    y = Variable{T}(- 𝝆 .* log.(𝒑), p.backprop)
    if y.backprop
        y.backward = function ∇CrossEntropy()
            if need2computeδ!(p)
                δ(p) .-= δ(y) .* 𝝆 ./ 𝒑
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
    𝒑  = ᵛ(p) .+ ϵ
    t₁ = @. -      𝝆  * log(    𝒑)
    t₂ = @. - (𝟙 - 𝝆) * log(𝟙 - 𝒑)
    y  = Variable{T}(t₁ + t₂, backprop)
    if backprop
        y.backward = function ∇BinaryCrossEntropy()
            if need2computeδ!(p)
                δ₁ = @. (𝟙 - 𝝆) / (𝟙 - 𝒑)
                δ₂ = @. 𝝆 / 𝒑
                δ(p) .+= δ(y) .* (δ₁ - δ₂)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    BinaryCrossEntropy(p::Variable, label::AbstractArray) -> y::Variable
binary cross entropy is `y = - label*log(p) - (1-label)*log(1-p)` where `p` is the output of the network.
"""
function BinaryCrossEntropy(p::Variable{T}, label::AbstractArray) where T
    @assert p.shape == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    𝝆  = label
    𝒑  = ᵛ(p) .+ ϵ
    t₁ = @. -      𝝆  * log(    𝒑)
    t₂ = @. - (𝟙 - 𝝆) * log(𝟙 - 𝒑)
    y  = Variable{T}(t₁ + t₂, p.backprop)
    if y.backprop
        y.backward = function ∇BinaryCrossEntropy()
            if need2computeδ!(p)
                δ₁ = @. (𝟙 - 𝝆) / (𝟙 - 𝒑)
                δ₂ = @. 𝝆 / 𝒑
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
    𝒑  = p + ϵ
    t₁ = @. -      label  * log(    𝒑)
    t₂ = @. - (𝟙 - label) * log(𝟙 - 𝒑)
    return t₁ + t₂
end


CrossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = Loss( CrossEntropy(x, label), reduction=reduction )
CrossEntropyLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = Loss( CrossEntropy(x, label), reduction=reduction )
CrossEntropyLoss(x::AbstractArray, label::AbstractArray; reduction::String="sum") = Loss( CrossEntropy(x, label), reduction=reduction )

BinaryCrossEntropyLoss(x::Variable{T}, label::Variable{T}; reduction::String="sum") where T = Loss(BinaryCrossEntropy(x, label), reduction=reduction)
BinaryCrossEntropyLoss(x::Variable{T}, label::AbstractArray; reduction::String="sum") where T = Loss(BinaryCrossEntropy(x, label), reduction=reduction)
BinaryCrossEntropyLoss(x::AbstractArray, label::AbstractArray; reduction::String="sum") = Loss(BinaryCrossEntropy(x, label), reduction=reduction)


"""
    FocalBCE(p::Variable, label::AbstractArray; focus::Real=1.0f0, alpha::Real=0.5f0)

focal loss version BinaryCrossEntropy
"""
function FocalBCE(p::Variable{T}, label::AbstractArray; focus::Real=1.0f0, alpha::Real=0.5f0) where T
    @assert p.shape == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    γ  = TO(focus)
    α  = TO(alpha)
    𝝆  = label
    𝒑  = ᵛ(p) .+ ϵ

    w₁ = @. -      α  *      𝝆
    w₂ = @. - (𝟙 - α) * (𝟙 - 𝝆)

    t₁ = @. w₁ * (𝟙 - 𝒑) ^ γ * log(    𝒑)
    t₂ = @. w₂ *      𝒑  ^ γ * log(𝟙 - 𝒑)

    y  = Variable{T}(t₁ + t₂, p.backprop)

    if y.backprop
        y.backward = function ∇FocalBCE()
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


"""
    FocalCE(p::Variable, label::AbstractArray; focus::Real=1.0f0)

focal loss version CrossEntropy
"""
function FocalCE(p::Variable{T}, label::AbstractArray; focus::Real=1.0f0) where T
    @assert p.shape == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    γ  = TO(focus)
    𝒑  = ᵛ(p) .+ ϵ
    𝝆  = label

    t = @. - 𝝆 * (𝟙 - 𝒑) ^ γ * log(𝒑)
    y = Variable{T}(t, p.backprop)

    if y.backprop
        y.backward = function ∇FocalCE()
            if need2computeδ!(p)
                δp = δ(p)
                δy = δ(y)
                @. δp += δy * 𝝆 * (𝟙 - 𝒑)^(γ - 𝟙) * (γ * log(𝒑) + 𝟙 - 𝟙 / 𝒑)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end

"""
    FocalCELoss(x::Variable,
                label::AbstractArray;
                focus::Real=1.0f0,
                reduction::String="sum")

focal loss version CrossEntropyLoss
"""
function FocalCELoss(x::Variable{T},
                     label::AbstractArray;
                     focus::Real=1.0f0,
                     reduction::String="sum") where T
    return Loss(FocalCE(x, label, focus=focus), reduction=reduction)
end


"""
    FocalBCELoss(x::Variable,
                 label::AbstractArray;
                 focus::Real=1.0f0,
                 alpha::Real=0.5f0,
                 reduction::String="sum")

focal loss version BinaryCrossEntropyLoss
"""
function FocalBCELoss(x::Variable{T},
                      label::AbstractArray;
                      focus::Real=1.0f0,
                      alpha::Real=0.5f0,
                      reduction::String="sum") where T
    return Loss(FocalBCE(x, label, focus=focus, alpha=alpha), reduction=reduction)
end


"""
    NLogCrossEntropy(p::Variable, label::AbstractArray)
Loss = [ − ln(`p`) ] * [ − `label` * ln(`p`) ], where `p` is the predicted probability
"""
function NLogCrossEntropy(p::Variable{T}, label::AbstractArray) where T
    # Loss = (-𝒍𝒏𝒑)*(-𝜸 * 𝒍𝒏𝒑), negative log weighted CELoss
    @assert p.shape == size(label)
    S = eltype(p)
    ϵ = S(1e-38)
    𝜸 = label
    𝒑 = ᵛ(p) .+ ϵ
    𝒍𝒏𝒑 = log.(𝒑)
    y = Variable{T}(𝜸 .* 𝒍𝒏𝒑 .* 𝒍𝒏𝒑, p.backprop)
    if y.backprop
        𝟚 = S(2f0)
        y.backward = function ∇NLogCrossEntropy()
            if need2computeδ!(p)
                δ(p) .+= δ(y) .* 𝟚 .* 𝜸 .* 𝒍𝒏𝒑 ./ 𝒑
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


function NLogCELoss(p::Variable, label::AbstractArray; reduction::String="sum")
    return Loss(NLogCrossEntropy(p, label), reduction=reduction)
end


"""
    InvPowerCrossEntropy(p::Variable{T}, label::AbstractArray; a::Real=0.3f0, n::Real=1f0)
Loss = [ 1 / (`p` + (1-`a`))`ⁿ` ] * [ − `label` * ln(`p`) ], where `p` is the predicted probability.
`a` in [0, 0.5] is recommended.
"""
function InvPowerCrossEntropy(p::Variable{T}, label::AbstractArray; a::Real=0.3f0, n::Real=1f0) where T
    @assert p.shape == size(label)
    S = eltype(p)
    ϵ = S(1e-38)
    a = S(1 - a)
    𝒏 = S(n)

    𝜸 = label
    𝒑 = ᵛ(p) .+ ϵ
    𝒍𝒏𝒑  = log.(𝒑)
    Q    = 𝒑 .+ a
    Qⁿ   = Q .^ 𝒏
    Qⁿ⁺¹ = Q .* Qⁿ
    y = Variable{T}( - 𝜸 .* 𝒍𝒏𝒑 ./ Qⁿ , p.backprop)

    if y.backprop
        y.backward = function ∇InvPowerCrossEntropy()
            if need2computeδ!(p)
                δy = δ(y)
                δp = δ(p)
                @. δp .+= δy * 𝜸 * (𝒏 * 𝒑 * 𝒍𝒏𝒑 - Q) / (𝒑 * Qⁿ⁺¹)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


function InvPowerCELoss(p::Variable,
                        label::AbstractArray;
                        reduction::String="sum",
                        a::Real=0.3f0,
                        n::Real=1.0f0)
    return Loss(InvPowerCrossEntropy(p, label, a=a, n=n), reduction=reduction)
end
