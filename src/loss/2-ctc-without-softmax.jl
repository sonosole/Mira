export DNNCTCLoss
export FNNCTCLoss
export RNNCTCLoss
export FRNNCTCLoss
export FRNNFocalCTCLoss
export FRNNCTCProbs

"""
    DNNCTCLoss(p::Variable{T}, seq; blank::Int=1, weight=1.0)

case batchsize==1 for test case, `p` here is probability or weighted probability

# Inputs
`p`      : 2-D Variable, probability or weighted probability\n
`seq`    : 1-D Array, input sequence's label\n
`weight` : weight for CTC loss

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function DNNCTCLoss(p::Variable{T}, seq; blank::Int=1, weight=1.0) where T
    r, loglikely = CTC(ᵛ(p), seq, blank=blank)
    y = Variable{T}([loglikely], p.backprop)

    if y.backprop
        y.backward = function DNNCTCLoss_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= δ(y) .* r ./ ᵛ(p)
                else
                    δ(p) .-= δ(y) .* r ./ ᵛ(p) .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    FNNCTCLoss(p::Variable, seqlabels::Vector, inputlens; blank::Int=1, weight=1.0)

# Inputs
`p`         : 2-D Variable, probability or weighted probability\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n
`weight`    : weight for CTC loss

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function FNNCTCLoss(p::Variable{T}, seqlabels::Vector, inputlens; blank::Int=1, weight=1.0) where T
    S = eltype(p)
    batchsize = length(inputLengths)
    loglikely = zeros(S, batchsize)
    I, F = indexbounds(inputlens)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        span = I[b]:F[b]
        r[:,span], loglikely[b] = CTC(p.value[:,span], seqlabels[b], blank=blank)
    end

    y = Variable{T}([sum(loglikely)], p.backprop)

    if y.backprop
        y.backward = function FNNCTCLoss_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= δ(y) .* r ./ ᵛ(p)
                else
                    δ(p) .-= δ(y) .* r ./ ᵛ(p) .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    RNNCTCLoss(p::Variable, seqlabels::Vector, inputlens; blank::Int=1, weight=1.0)

# Inputs
`p`         : 3-D Variable with shape (featdims,timesteps,batchsize), probability or weighted probability\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : each input's length, like [19,97,...]\n
`weight`    : weight for CTC loss

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function RNNCTCLoss(p::Variable{T},
                    seqlabels::Vector,
                    inputlens;
                    blank::Int=1,
                    weight=1.0,
                    reduction::String="seqlen") where T
    S = eltype(p)
    batchsize = length(inputlens)
    loglikely = zeros(S, batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        Tᵇ = inputlens[b]
        r[:,1:Tᵇ,b], loglikely[b] = CTC(p.value[:,1:Tᵇ,b], seqlabels[b], blank=blank)
    end

    reduce3d(r, loglikely, seqlabels, reduction)
    y = Variable{T}([sum(loglikely)], p.backprop)

    if y.backprop
        y.backward = function RNNCTCLoss_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= δ(y) .* r ./ ᵛ(p)
                else
                    δ(p) .-= δ(y) .* r ./ ᵛ(p) .* S(weight)
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    FRNNCTCLoss(p::Variable,
                seqlabels::Vector;
                blank::Int=1,
                weight=1.0,
                reduction::String="seqlen")

# Inputs
`p`         : 3-D Variable with shape (featdims,timesteps,batchsize), probability or weighted probability\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`weight`    : weight for CTC loss

# Structure
                   ┌───┐
                   │ │ │
                   │ W ├──►──┐
                   │ │ │     │
                   └───┘     │
    ┌───┐          ┌───┐     │     ┌───┐
    │ │ │ softmax  │ │ │   ┌─┴─┐   │ │ │   ┌───────┐
    │ X ├─────────►│ Y ├──►│ × ├──►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┘   │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘           └───┘       │
      │              │+                        ▼
    ┌─┴─┐            ▼                       ┌─┴─┐
    │ │ │          ┌─┴─┐ -                   │ │ │
    │ δ │◄─────────┤ - │─────────◄───────────┤ r │
    │ │ │          └───┘                     │ │ │
    └───┘                                    └───┘
"""
function FRNNCTCLoss(p::Variable{T},
                     seqlabels::Vector;
                     blank::Int=1,
                     weight=1.0,
                     reduction::String="seqlen") where T
    S = eltype(p)
    featdims, timesteps, batchsize = size(p)
    loglikely = zeros(S, batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = CTC(p.value[:,:,b], seqlabels[b], blank=blank)
    end

    reduce3d(r, loglikely, seqlabels, reduction)
    y = Variable{T}([sum(loglikely)], p.backprop)

    if y.backprop
        y.backward = function FRNNCTCLoss_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= δ(y) .* r ./ ᵛ(p)
                else
                    δ(p) .-= δ(y) .* r ./ ᵛ(p) .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


"""
    FRNNFocalCTCLoss(p::Variable,
                     seqlabels::Vector;
                     blank::Int=1,
                     gamma=2,
                     weight::Float64=1.0,
                     reduction="seqlen")

# Inputs
`p`         : 3-D Variable with shape (featdims,timesteps,batchsize), probability\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`weight`    : weight for CTC loss

# Structure

    ┌───┐          ┌───┐
    │ │ │ softmax  │ │ │   ┌─────────────┐
    │ X ├─────────►│ P ├──►│Focal CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └─────────────┘
    └───┘          └───┘
"""
function FRNNFocalCTCLoss(p::Variable{T},
                          seqlabels::Vector;
                          blank::Int=1,
                          gamma::Real=2,
                          weight=1.0,
                          reduction::String="seqlen") where T

    S = eltype(p)
    featdims, timesteps, batchsize = size(p)
    loglikely = zeros(S, 1, 1, batchsize)
    r = zero(ᵛ(p))
    𝜸 = S(gamma)
    𝟙 = S(1.0f0)

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = CTC(p.value[:,:,b], seqlabels[b], blank=blank)
    end

    𝒍𝒏𝒑 = T(-loglikely)
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


# naive implementation, more ops needed, good for learning
function FRNNFocalCTCLoss_Naive(p::Variable{T},
                                seqlabels::Vector;
                                blank::Int=1,
                                gamma::Real=2,
                                weight=1.0,
                                reduction::String="seqlen") where T
    featdims, timesteps, batchsize = size(p)
    S = eltype(p)
    loglikely = zeros(S, 1, 1, batchsize)
    r = zero(ᵛ(p))
    𝜸 = S(gamma)
    𝟙 = S(1.0f0)

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = CTC(p.value[:,:,b], seqlabels[b], blank=blank)
    end

    𝒍𝒏𝒑 = T(-loglikely)
    𝒑 = Variable{T}(exp(𝒍𝒏𝒑), p.backprop)
    y = (-(1 - 𝒑)^𝜸) .* log(𝒑)
    reduce3d(r, ᵛ(y), seqlabels, reduction)

    if 𝒑.backprop
        𝒑.backward = function FRNNFocalCTCLoss_Naive_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .+= δ(𝒑) .* ᵛ(𝒑) .* r ./ ᵛ(p)
                else
                    δ(p) .+= δ(𝒑) .* ᵛ(𝒑) .* r ./ ᵛ(p) .* weight
                end
            end
            ifNotKeepδThenFreeδ!(𝒑)
        end
        addchild(𝒑, p)
    end
    return loss(y)
end


"""
    FRNNCTCProbs(p::Variable, seqlabels::Vector; blank::Int=1) -> prob::Variable

# Inputs
`p`         : 3-D Variable (featdims,timesteps,batchsize), output of softmax\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`weight`    : weight for CTC loss

# Output
`prob`      : 3-D Variable (1,1,batchsize), i.e. `prob` is the probabilities of each sequence
"""
function FRNNCTCProbs(p::Variable{T}, seqlabels::Vector; blank::Int=1) where T
    featdims, timesteps, batchsize = size(p)
    loglikely = zeros(eltype(p), batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = CTC(p.value[:,:,b], seqlabels[b], blank=blank)
    end

    𝒑 = Variable{T}(exp(T(-loglikely)), x.backprop)

    if 𝒑.backprop
        𝒑.backward = function FRNNCTCProbs_Backward()
            if need2computeδ!(p)
                δ(p) .-= δ(𝒑) .* r ./ ᵛ(p)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(𝒑, p)
    end
    return 𝒑
end
