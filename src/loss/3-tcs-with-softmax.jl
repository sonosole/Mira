export FNNSoftmaxTCSLoss
export RNNSoftmaxTCSLoss
export FRNNSoftmaxTCSLoss
export FRNNSoftmaxFocalTCSLoss
export FRNNSoftmaxTCSProbs

"""
    FNNSoftmaxTCSLoss(x::Variable,
                      seqlabels::VecVecInt,
                      inputlens::VecInt;
                      background::Int=1,
                      foreground::Int=2)

# Inputs
`x`         : 2-D Variable, a batch of concatenated input sequence\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n

# Structure
    ┌───┐
    │ │ │
    │ W ├──►─┐
    │ │ │    │
    └───┘    │
    ┌───┐    │    ┌───┐          ┌───┐
    │ │ │  ┌─┴─┐  │ │ │ softmax  │ │ │   ┌───────┐
    │ Z ├─►│ × ├─►│ X ├─────────►│ P ├──►│TCSLOSS│◄── (seqLabel)
    │ │ │  └───┘  │ │ │          │ │ │   └───┬───┘
    └───┘         └─┬─┘          └─┬─┘       │
                    │              │+        ▼
                  ┌─┴─┐            ▼       ┌─┴─┐
                  │ │ │          ┌─┴─┐ -   │ │ │
                  │ δ │◄─────────┤ - │──◄──┤ r │
                  │ │ │          └───┘     │ │ │
                  └───┘                    └───┘
"""
function FNNSoftmaxTCSLoss(x::Variable{T},
                           seqlabels::VecVecInt,
                           inputlens::VecInt;
                           background::Int=1,
                           foreground::Int=2) where T
    batchsize = length(seqlabels)
    nlnp = zeros(eltype(x), batchsize)
    I, F = indexbounds(inputlens)
    p = softmax(ᵛ(x); dims=1)
    r = zero(p)

    for b = 1:batchsize
        span = I[b]:F[b]
        r[:,span], nlnp[b] = TCS(p[:,span], seqlabels[b], background=background, foreground=foreground)
    end

    Δ = p - r
    y = Variable{T}([sum(nlnp)], x.backprop)

    if y.backprop
        y.backward = function ∇FNNSoftmaxTCSLoss()
            if needgrad(x)
                x ← δ(y) .* Δ
            end
        end
        addchild(y, x)
    end
    return y
end


"""
    RNNSoftmaxTCSLoss(x::Variable,
                      seqlabels::VecVecInt,
                      inputlens::VecInt;
                      reduction::String="seqlen",
                      background::Int=1,
                      foreground::Int=2)

a batch of padded input sequence is processed by neural networks into `x`

# Inputs
`x`         : 3-D Variable with shape (featdims,timesteps,batchsize), a batch of padded input sequence\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n

# Structure
    ┌───┐
    │ │ │
    │ W ├──►─┐
    │ │ │    │
    └───┘    │
    ┌───┐    │    ┌───┐          ┌───┐
    │ │ │  ┌─┴─┐  │ │ │ softmax  │ │ │   ┌───────┐
    │ Z ├─►│ × ├─►│ X ├─────────►│ P ├──►│TCSLOSS│◄── (seqLabel)
    │ │ │  └───┘  │ │ │          │ │ │   └───┬───┘
    └───┘         └─┬─┘          └─┬─┘       │
                    │              │+        ▼
                  ┌─┴─┐            ▼       ┌─┴─┐
                  │ │ │          ┌─┴─┐ -   │ │ │
                  │ δ │◄─────────┤ - │──◄──┤ r │
                  │ │ │          └───┘     │ │ │
                  └───┘                    └───┘
"""
function RNNSoftmaxTCSLoss(x::Variable{T},
                           seqlabels::VecVecInt,
                           inputlens::VecInt;
                           reduction::String="seqlen",
                           background::Int=1,
                           foreground::Int=2) where T
    batchsize = length(seqlabels)
    nlnp = zeros(eltype(x), 1, 1, batchsize)
    p = zero(ᵛ(x))
    r = zero(p)

    for b = 1:batchsize
        Tᵇ = inputlens[b]
        p[:,1:Tᵇ,b] = softmax(x.value[:,1:Tᵇ,b]; dims=1)
        r[:,1:Tᵇ,b], nlnp[b] = TCS(p[:,1:Tᵇ,b], seqlabels[b], background=background, foreground=foreground)
    end

    Δ = p - r
    l = T(nlnp)
    reduce3d(Δ, l, seqlabels, reduction)
    y = Variable{T}([sum(l)], x.backprop)

    if y.backprop
        y.backward = function ∇RNNSoftmaxTCSLoss()
            if needgrad(x)
                x ← δ(y) .* Δ
            end
        end
        addchild(y, x)
    end
    return y
end


"""
    FRNNSoftmaxTCSLoss(x::Variable,
                       seqlabels::VecVecInt;
                       reduction::String="seqlen",
                       background::Int=1,
                       foreground::Int=2)

# Main Inputs
`x`            : 3-D Variable with shape (featdims,timesteps,batchsize), resulted by a batch of padded input sequence\n
`seqlabels`    : a batch of sequential labels, like [[i,j,k],[x,y],...]\n

# Structure
    ┌───┐
    │ │ │
    │ W ├──►─┐
    │ │ │    │
    └───┘    │
    ┌───┐    │    ┌───┐          ┌───┐
    │ │ │  ┌─┴─┐  │ │ │ softmax  │ │ │   ┌───────┐
    │ Z ├─►│ × ├─►│ X ├─────────►│ P ├──►│TCSLOSS│◄── (seqLabel)
    │ │ │  └───┘  │ │ │          │ │ │   └───┬───┘
    └───┘         └─┬─┘          └─┬─┘       │
                    │              │+        ▼
                  ┌─┴─┐            ▼       ┌─┴─┐
                  │ │ │          ┌─┴─┐ -   │ │ │
                  │ δ │◄─────────┤ - │──◄──┤ r │
                  │ │ │          └───┘     │ │ │
                  └───┘                    └───┘
"""
function FRNNSoftmaxTCSLoss(x::Variable{T},
                            seqlabels::VecVecInt;
                            reduction::String="seqlen",
                            background::Int=1,
                            foreground::Int=2) where T
    featdims, timesteps, batchsize = size(x)
    nlnp = zeros(eltype(x), 1, 1, batchsize)
    p = softmax(ᵛ(x); dims=1)
    r = zero(p)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = TCS(p[:,:,b], seqlabels[b], background=background, foreground=foreground)
    end

    Δ = p - r
    l = T(nlnp)
    reduce3d(Δ, l, seqlabels, reduction)
    y = Variable{T}([sum(l)], x.backprop)

    if y.backprop
        y.backward = function ∇FRNNSoftmaxTCSLoss()
            if needgrad(x)
                x ← δ(y) .* Δ
            end
        end
        addchild(y, x)
    end
    return y
end


function FRNNSoftmaxTCSProbs(x::Variable{T},
                             seqlabels::VecVecInt;
                             background::Int=1,
                             foreground::Int=2) where T
    featdims, timesteps, batchsize = size(x)
    nlnp = zeros(eltype(x), 1, 1, batchsize)
    p = softmax(ᵛ(x); dims=1)
    r = zero(p)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = TCS(p[:,:,b], seqlabels[b], background=background, foreground=foreground)
    end

    𝒑 = Variable{T}(exp(T(-nlnp)), x.backprop)
    Δ = r - p

    if 𝒑.backprop
        𝒑.backward = function ∇FRNNSoftmaxCTCProbs()
            if needgrad(x)
                x ← δ(𝒑)  .* ᵛ(𝒑) .* Δ
            end
        end
        addchild(𝒑, x)
    end
    return 𝒑
end


function FRNNSoftmaxFocalTCSLoss(x::Variable{T},
                                 seqlabels::VecVecInt;
                                 reduction::String="seqlen",
                                 background::Int=1,
                                 foreground::Int=2,
                                 focus::Real=1.0f0) where T
    featdims, timesteps, batchsize = size(x)
    S = eltype(x)
    nlnp = zeros(S, 1, 1, batchsize)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)
    𝜸 = S(focus)
    𝟙 = S(1.0f0)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = TCS(p[:,:,b], seqlabels[b], background=background, foreground=foreground)
    end

    𝒍𝒏𝒑 = T(-nlnp)
    𝒑 = exp(𝒍𝒏𝒑)
    𝒌 = @.  (𝟙 - 𝒑)^(𝜸-𝟙) * (𝟙 - 𝒑 - 𝜸*𝒑*𝒍𝒏𝒑)
    t = @. -(𝟙 - 𝒑)^𝜸 * 𝒍𝒏𝒑
    Δ = p - r
    reduce3d(Δ, t, seqlabels, reduction)
    y = Variable{T}([sum(t)], x.backprop)

    if y.backprop
        y.backward = function ∇FRNNSoftmaxFocalTCSLoss()
            if needgrad(x)
                x ← δ(y) .* 𝒌 .* Δ
            end
        end
        addchild(y, x)
    end
    return y
end
