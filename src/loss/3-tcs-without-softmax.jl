export DNN_Batch_TCS
export RNN_Batch_TCS
export CRNN_Batch_TCS


"""
    DNN_Batch_TCS(p::Variable{T},
                  seqlabels::Vector,
                  inputlens;
                  background::Int=1,
                  foreground::Int=2,
                  reduction::String="seqlen"
                  weight=1.0) where T

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
function DNN_Batch_TCS(p::Variable{T},
                       seqlabels::Vector,
                       inputlens;
                       background::Int=1,
                       foreground::Int=2,
                       reduction::String="seqlen"
                       weight=1.0) where T
    S = eltype(p)
    batchsize = length(seqlabels)
    loglikely = zeros(S, batchsize)
    I, F = indexbounds(inputlens)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        span = I[b]:F[b]
        r[:,span], loglikely[b] = TCS(p.value[:,span], seqlabels[b], background=background, foreground=foreground)
    end

    reduce3d(r, loglikely, seqlabels, reduction)
    y = Variable{T}([sum(loglikely)/batchsize], p.backprop)

    if y.backprop
        y.backward = function DNN_Batch_TCS_Backward()
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


"""
    RNN_Batch_TCS(p::Variable{T},
                  seqlabels::Vector,
                  inputlens;
                  background::Int=1,
                  foreground::Int=2,
                  reduction::String="seqlen"
                  weight=1.0) where T

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
function RNN_Batch_TCS(p::Variable{T},
                       seqlabels::Vector,
                       inputlens;
                       background::Int=1,
                       foreground::Int=2,
                       reduction::String="seqlen"
                       weight=1.0) where T
    S = eltype(p)
    batchsize = length(seqlabels)
    loglikely = zeros(S, batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        Tᵇ = inputlens[b]
        Lᵇ = length(seqlabels[b])
        r[:,1:Tᵇ,b], loglikely[b] = TCS(p.value[:,1:Tᵇ,b], seqlabels[b], background=background, foreground=foreground)
    end

    reduce3d(r, loglikely, seqlabels, reduction)
    y = Variable{T}([sum(loglikely)], p.backprop)

    if y.backprop
        y.backward = function RNN_Batch_TCS_Backward()
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

"""
    CRNN_Batch_TCS(p::Variable{T},
                   seqlabels::Vector;
                   background::Int=1,
                   foreground::Int=2,
                   reduction::String="seqlen"
                   weight=1.0) where T

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
function CRNN_Batch_TCS(p::Variable{T},
                        seqlabels::Vector;
                        background::Int=1,
                        foreground::Int=2,
                        reduction::String="seqlen"
                        weight=1.0) where T
    S = eltype(p)
    featdims, timesteps, batchsize = size(p)
    loglikely = zeros(S, batchsize)
    r = zero(ᵛ(p))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = TCS(p.value[:,:,b], seqlabels[b], background=background, foreground=foreground)
    end

    reduce3d(r, loglikely, seqlabels, reduction)
    y = Variable{T}([sum(loglikely)], p.backprop)

    if y.backprop
        y.backward = function CRNN_Batch_TCS_Backward()
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
