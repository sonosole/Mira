export DNN_CTC
export DNN_Batch_CTC
export RNN_Batch_CTC
export CRNN_Batch_CTC


"""
    DNN_CTC(p::Variable{T}, seq; blank=1, weight=1.0)

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
function DNN_CTC(p::Variable{T}, seq; blank=1, weight=1.0) where T
    L = length(seq) * 2 + 1
    r, loglikely = CTC(ᵛ(p), seq, blank=blank)
    y = Variable{T}([loglikely / L], p.backprop)

    if y.backprop
        y.backward = function DNN_CTC_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T))
                else
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T)) .* weight
                end
            end
        end
        addchild(y, x)
    end
    return y
end


"""
    DNN_Batch_CTC(p::Variable{T}, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T

a batch of concatenated input sequence is processed by neural networks into `p`

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
function DNN_Batch_CTC(p::Variable{T}, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T
    batchsize = length(inputLengths)
    loglikely = zeros(eltype(x), batchsize)
    I, F = indexbounds(inputlens)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        span = I[b]:F[b]
        r[:,span], loglikely[b] = CTC(p.value[:,span], seqlabels[b], blank=blank)
        loglikely[b] /= length(seqlabels[b]) * 2 + 1
    end

    y = Variable{T}([sum(loglikely)/batchsize], x.backprop)
    if y.backprop
        y.backward = function DNN_Batch_CTC_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T))
                else
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T)) .* weight
                end
            end
        end
        addchild(y, x)
    end
    return y
end


"""
    RNN_Batch_CTC(p::Variable{T}, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T

a batch of padded input sequence is processed by neural networks into `p`

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
function RNN_Batch_CTC(p::Variable{T}, seqlabels::Vector, inputlens; blank=1, weight=1.0) where T
    batchsize = length(inputlens)
    loglikely = zeros(eltype(x), batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        Tᵇ = inputlens[b]
        Lᵇ = length(seqlabels[b])
        r[:,1:Tᵇ,b], loglikely[b] = CTC(p.value[:,1:Tᵇ,b], seqlabels[b], blank=blank)
        loglikely[b] /= Lᵇ * 2 + 1
    end

    y = Variable{T}([sum(loglikely)/batchsize], x.backprop)
    if y.backprop
        y.backward = function RNN_Batch_CTC_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T))
                else
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T)) .* weight
                end
            end
        end
        addchild(y, x)
    end
    return y
end


"""
    CRNN_Batch_CTC(p::Variable{T}, seqlabels::Vector) where T -> LogLikely

a batch of padded input sequence is processed by neural networks into `p`

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
function CRNN_Batch_CTC(p::Variable{T}, seqlabels::Vector; blank=1, weight=1.0) where T
    featdims, timesteps, batchsize = size(p)
    loglikely = zeros(eltype(x), batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = CTC(p.value[:,:,b], seqlabels[b], blank=blank)
        loglikely[b] /= length(seqlabels[b]) * 2 + 1
    end

    y = Variable{T}([sum(loglikely)/batchsize], x.backprop)
    if y.backprop
        y.backward = function CRNN_Batch_CTC_Backward()
            if need2computeδ!(p)
                if weight==1.0
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T))
                else
                    δ(p) .-= r ./ (ᵛ(p) .+ eps(T)) .* weight
                end
            end
        end
        addchild(y, x)
    end
    return y
end


"""
    CRNN_Focal_CTC(p::Variable{T}, seqlabels::Vector; blank=1, gamma=2, reduction="seqlen")

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
function CRNN_Focal_CTC(p::Variable{T}, seqlabels::Vector; blank=1, gamma=2, reduction="seqlen") where T
    featdims, timesteps, batchsize = size(p)
    loglikely = zeros(eltype(x), batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], _ = CTC(p.value[:,:,b], seqlabels[b], blank=blank)
    end
    y = seqfocalCE(p, r, seqlabels, gamma=gamma, reduction=reduction)
    return loss(y)
end
