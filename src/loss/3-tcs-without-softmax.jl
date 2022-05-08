export FNNTCSLoss
export RNNTCSLoss
export FRNNTCSLoss
export FRNNTCSProbs
export FRNNFocalTCSLoss

"""
    FNNTCSLoss(p::Variable,
               seqlabels::Vector,
               inputlens;
               background::Int=1,
               foreground::Int=2,
               weight=1.0)

a batch of concatenated input sequence is processed by neural networks into `p`

# Inputs
`p`         : 2-D Variable, probability or weighted probability\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n
`weight`    : weight for TCS loss

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│TCSLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function FNNTCSLoss(p::Variable{T},
                    seqlabels::Vector,
                    inputlens;
                    background::Int=1,
                    foreground::Int=2,
                    weight=1.0) where T
    batchsize = length(seqlabels)
    nlnp = zeros(S, batchsize)
    I, F = indexbounds(inputlens)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        span = I[b]:F[b]
        r[:,span], nlnp[b] = TCS(p.value[:,span], seqlabels[b], background=background, foreground=foreground)
    end

    y = Variable{T}([sum(nlnp)], p.backprop)

    if y.backprop
        y.backward = function FNNTCSLoss_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ ᵛ(p)
                else
                    δ(p) .-= r ./ ᵛ(p) .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    RNNTCSLoss(p::Variable,
               seqlabels::Vector,
               inputlens;
               background::Int=1,
               foreground::Int=2,
               reduction::String="seqlen",
               weight=1.0)

a batch of padded input sequence is processed by neural networks into `p`

# Inputs
`p`         : 3-D Variable with shape (featdims,timesteps,batchsize), probability or weighted probability\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n
`weight`    : weight for TCS loss

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│TCSLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function RNNTCSLoss(p::Variable{T},
                    seqlabels::Vector,
                    inputlens;
                    background::Int=1,
                    foreground::Int=2,
                    reduction::String="seqlen",
                    weight=1.0) where T
    S = eltype(p)
    batchsize = length(seqlabels)
    nlnp = zeros(S, batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        Tᵇ = inputlens[b]
        r[:,1:Tᵇ,b], nlnp[b] = TCS(p.value[:,1:Tᵇ,b], seqlabels[b], background=background, foreground=foreground)
    end

    reduce3d(r, nlnp, seqlabels, reduction)
    y = Variable{T}([sum(nlnp)], p.backprop)

    if y.backprop
        y.backward = function RNNTCSLoss_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ ᵛ(p)
                else
                    δ(p) .-= r ./ ᵛ(p) .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end

"""
    FRNNTCSLoss(p::Variable,
                seqlabels::Vector;
                background::Int=1,
                foreground::Int=2,
                reduction::String="seqlen",
                weight=1.0)

a batch of padded input sequence is processed by neural networks into `p`

# Main Inputs
`p`            : 3-D Variable with shape (featdims,timesteps,batchsize), probability or weighted probability\n
`seqlabels`    : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`weight`       : weight for TCS loss

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│TCSLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function FRNNTCSLoss(p::Variable{T},
                     seqlabels::Vector;
                     background::Int=1,
                     foreground::Int=2,
                     reduction::String="seqlen",
                     weight=1.0) where T
    S = eltype(p)
    featdims, timesteps, batchsize = size(p)
    nlnp = zeros(S, batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], nlnp[b] = TCS(p.value[:,:,b], seqlabels[b], background=background, foreground=foreground)
    end

    reduce3d(r, nlnp, seqlabels, reduction)
    y = Variable{T}([sum(nlnp)], p.backprop)

    if y.backprop
        y.backward = function FRNNTCSLoss_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ ᵛ(p)
                else
                    δ(p) .-= r ./ ᵛ(p) .* S(weight)
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


function FRNNFocalTCSLoss(p::Variable{T},
                          seqlabels::Vector;
                          background::Int=1,
                          foreground::Int=2,
                          reduction::String="seqlen",
                          gamma::Real=2,
                          weight=1.0) where T
    S = eltype(p)
    featdims, timesteps, batchsize = size(p)
    nlnp = zeros(S, 1, 1, batchsize)
    r = zero(ᵛ(p))
    𝜸 = S(gamma)
    𝟙 = S(1.0f0)

    Threads.@threads for b = 1:batchsize
        r[:,:,b], nlnp[b] = TCS(p.value[:,:,b], seqlabels[b], background=background, foreground=foreground)
    end

    𝒍𝒏𝒑 = T(-nlnp)
    𝒑 = exp(𝒍𝒏𝒑)
    𝒌 = @.  (𝟙 - 𝒑)^(𝜸-𝟙) * (𝜸*𝒑*𝒍𝒏𝒑 + 𝒑 - 𝟙)
    t = @. -(𝟙 - 𝒑)^𝜸 * 𝒍𝒏𝒑

    reduce3d(r, t, seqlabels, reduction)
    y = Variable{T}([sum(t)], p.backprop)

    if y.backprop
        y.backward = function FRNNFocalCTCLoss_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .+= δ(y) .* 𝒌 .* r ./ ᵛ(p)
                else
                    δ(p) .+= δ(y) .* 𝒌 .* r ./ ᵛ(p) .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


function FRNNTCSProbs(p::Variable{T}, seqlabels::Vector; background::Int=1, foreground::Int=2) where T
    S = eltype(p)
    featdims, timesteps, batchsize = size(p)
    nlnp = zeros(S, 1, 1, batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], nlnp[b] = TCS(p.value[:,:,b], seqlabels[b], background=background, foreground=foreground)
    end

    𝒑 = Variable{T}(exp(T(-nlnp)), p.backprop)

    if 𝒑.backprop
        𝒑.backward = function FRNNTCSProbs_Backward()
            if need2computeδ!(p)
                δ(p) .+= δ(𝒑) .* ᵛ(𝒑) .* r ./ ᵛ(p)
            end
            ifNotKeepδThenFreeδ!(𝒑)
        end
        addchild(𝒑, p)
    end
    return 𝒑
end
