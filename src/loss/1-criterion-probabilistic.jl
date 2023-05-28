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
    ϵ = eltype(p)(1e-38)
    𝝆 = value(label)
    𝒑 = value(p) .+ ϵ

    y = Variable{T}(- 𝝆 .* log.(𝒑), backprop)
    if backprop
        y.backward = function ∇CrossEntropy()
            if need2computeδ!(p)
                p ← - δ(y) .* 𝝆 ./ 𝒑
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
    ϵ = eltype(p)(1e-38)
    𝒑 = value(p) .+ ϵ
    𝝆 = label

    y = Variable{T}(- 𝝆 .* log.(𝒑), p.backprop)
    if y.backprop
        y.backward = function ∇CrossEntropy()
            if need2computeδ!(p)
                p ← - δ(y) .* 𝝆 ./ 𝒑
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
    l  = TO(1.0f0)
    𝝆  = ᵛ(label)
    𝒑  = ᵛ(p)
    p⁺ = 𝒑 .+ ϵ
    p⁻ = 𝒑 .- ϵ

    t₁ = @. -      𝝆  * log(    p⁺)
    t₂ = @. - (l - 𝝆) * log(l - p⁻)
    y  = Variable{T}(t₁ + t₂, backprop)
    if backprop
        y.backward = function ∇BinaryCrossEntropy()
            if need2computeδ!(p)
                δ₁ = @. (l - 𝝆) / (l - p⁻)
                δ₂ = @.      𝝆  /      p⁺
                p ← δ(y) .* (δ₁ - δ₂)
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
    l  = TO(1.0f0)
    𝝆  = label
    𝒑  = ᵛ(p)
    p⁺ = 𝒑 .+ ϵ
    p⁻ = 𝒑 .- ϵ

    t₁ = @. -      𝝆  * log(    p⁺)
    t₂ = @. - (l - 𝝆) * log(l - p⁻)
    y  = Variable{T}(t₁ + t₂, p.backprop)
    if y.backprop
        y.backward = function ∇BinaryCrossEntropy()
            if need2computeδ!(p)
                δ₁ = @. (l - 𝝆) / (l - p⁻)
                δ₂ = @.      𝝆  /      p⁺
                p ← δ(y) .* (δ₁ - δ₂)
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
    l  = TO(1.0f0)
    t₁ = @. -      label  * log(    p + ϵ)
    t₂ = @. - (l - label) * log(l - p + ϵ)
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

focal loss version of BinaryCrossEntropy:\n
`loss = α * (1-p)ᵞ * [- label * ln(p)] + (1 - α) * pᵞ * [-(1-label) * ln(1-p)]`\n
where `γ` is the `focus` value, `α` is the weight for positive class.
"""
function FocalBCE(p::Variable{T}, label::AbstractArray; focus::Real=1.0f0, alpha::Real=0.5f0) where T
    @assert p.shape == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)
    l  = TO(1.0f0)
    γ  = TO(focus)
    α  = TO(alpha)
    𝝆  = label
    𝒑  = ᵛ(p)
    p⁺ = 𝒑 .+ ϵ
    p⁻ = 𝒑 .- ϵ

    w₁ = @. -      α  *      𝝆
    w₂ = @. - (l - α) * (l - 𝝆)

    t₁ = @. w₁ * (l - p⁻) ^ γ * log(    p⁺)
    t₂ = @. w₂ *      p⁺  ^ γ * log(l - p⁻)

    y  = Variable{T}(t₁ + t₂, p.backprop)

    if y.backprop
        y.backward = function ∇FocalBCE()
            if need2computeδ!(p)
                δ₁ = @. w₁ * (l - p⁻)^(γ - l) * (l / p⁺ - γ * log(p⁺) - l)
                δ₂ = @. w₂ * p⁺ ^ γ * (l / (p⁻ - l) + γ * log(l - p⁻) / p⁺)
                p ← δ(y) .* (δ₁ + δ₂)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    FocalCE(p::Variable, label::AbstractArray; focus::Real=1.0f0) -> y::Variable

focal loss version of CrossEntropy:\n
`loss = (1-p)ᵞ * [- label * ln(p)]`, \n
where `γ` is the `focus` value.
"""
function FocalCE(p::Variable{T}, label::AbstractArray; focus::Real=1.0f0) where T
    @assert p.shape == size(label)
    TO = eltype(p)
    ϵ  = TO(1e-38)  # alias for value closing to zero
    l  = TO(1.0f0)  # alias for value one
    γ  = TO(focus)
    𝝆  = label
    𝒑  = value(p)
    p⁺ = 𝒑 .+ ϵ    # little greater
    p⁻ = 𝒑 .- ϵ    # little smaller

    t = @. (l - p⁻) ^ γ * (- 𝝆 * log(p⁺))
    y = Variable{T}(t, p.backprop)

    if y.backprop
        y.backward = function ∇FocalCE()
            if need2computeδ!(p)
                n  = γ - l
                δp = δ(p)
                δy = δ(y)
                p  ← @. δy * 𝝆 * (l - p⁻)^n * (γ * log(p⁺) + l - l / p⁺)
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
        𝟐 = S(2f0)
        y.backward = function ∇NLogCrossEntropy()
            if need2computeδ!(p)
                p ← δ(y) .* 𝟐 .* 𝜸 .* 𝒍𝒏𝒑 ./ 𝒑
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
                p  ← @. δy * 𝜸 * (𝒏 * 𝒑 * 𝒍𝒏𝒑 - Q) / (𝒑 * Qⁿ⁺¹)
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
