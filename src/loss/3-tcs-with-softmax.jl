export DNN_Batch_TCS_With_Softmax
export RNN_Batch_TCS_With_Softmax
export CRNN_Batch_TCS_With_Softmax


"""
    DNN_Batch_TCS_With_Softmax(x::Variable{T},
                               seqlabels::Vector,
                               inputlens;
                               background::Int=1,
                               foreground::Int=2,
                               reduction::String="seqlen",
                               weight=1.0) where T

a batch of concatenated input sequence is processed by neural networks into `x`

# Inputs
`x`         : 2-D Variable, a batch of concatenated input sequence\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n
`weight`    : weight for TCS loss

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
function DNN_Batch_TCS_With_Softmax(x::Variable{T},
                                    seqlabels::Vector,
                                    inputlens;
                                    background::Int=1,
                                    foreground::Int=2,
                                    reduction::String="seqlen",
                                    weight=1.0) where T
    batchsize = length(seqlabels)
    loglikely = zeros(eltype(x), batchsize)
    I, F = indexbounds(inputlens)
    p = softmax(ᵛ(x); dims=1)
    r = zero(ᵛ(x))

    Threads.@threads for b = 1:batchsize
        span = I[b]:F[b]
        r[:,span], loglikely[b] = TCS(p[:,span], seqlabels[b], background=background, foreground=foreground)
    end

    Δ = p - r
    reduce3d(Δ, loglikely, seqlabels, reduction)
    y = Variable{T}([sum(loglikely)], x.backprop)

    if y.backprop
        y.backward = function DNN_Batch_TCS_With_Softmax_Backward()
            if need2computeδ!(x)
                if weight==1.0
                    δ(x) .+= δ(y) .* Δ
                else
                    δ(x) .+= δ(y) .* Δ .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    RNN_Batch_TCS_With_Softmax(x::Variable{T},
                               seqlabels::Vector,
                               inputlens;
                               background::Int=1,
                               foreground::Int=2,
                               reduction::String="seqlen",
                               weight=1.0) where T

a batch of padded input sequence is processed by neural networks into `x`

# Inputs
`x`         : 3-D Variable with shape (featdims,timesteps,batchsize), a batch of padded input sequence\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n
`weight`    : weight for TCS loss

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
function RNN_Batch_TCS_With_Softmax(x::Variable{T},
                                    seqlabels::Vector,
                                    inputlens;
                                    background::Int=1,
                                    foreground::Int=2,
                                    reduction::String="seqlen",
                                    weight=1.0) where T
    batchsize = length(seqlabels)
    loglikely = zeros(eltype(x), batchsize)
    p = zero(ᵛ(x))
    r = zero(ᵛ(x))

    Threads.@threads for b = 1:batchsize
        Tᵇ = inputlens[b]
        Lᵇ = length(seqlabels[b])
        p[:,1:Tᵇ,b] = softmax(x.value[:,1:Tᵇ,b]; dims=1)
        r[:,1:Tᵇ,b], loglikely[b] = TCS(p[:,1:Tᵇ,b], seqlabels[b], background=background, foreground=foreground)
    end

    Δ = p - r
    reduce3d(Δ, loglikely, seqlabels, reduction)
    y = Variable{T}([sum(loglikely)], x.backprop)

    if y.backprop
        y.backward = function RNN_Batch_TCS_With_Softmax_Backward()
            if need2computeδ!(x)
                if weight==1.0
                    δ(x) .+= δ(y) .* Δ
                else
                    δ(x) .+= δ(y) .* Δ .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

"""
    CRNN_Batch_TCS_With_Softmax(x::Variable{T},
                                seqlabels::Vector;
                                background::Int=1,
                                foreground::Int=2,
                                reduction::String="seqlen",
                                weight=1.0) where T

a batch of padded input sequence is processed by neural networks into `x`

# Main Inputs
`x`            : 3-D Variable with shape (featdims,timesteps,batchsize), resulted by a batch of padded input sequence\n
`seqlabels`    : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`weight`       : weight for TCS loss

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
function CRNN_Batch_TCS_With_Softmax(x::Variable{T},
                                     seqlabels::Vector;
                                     background::Int=1,
                                     foreground::Int=2,
                                     reduction::String="seqlen",
                                     weight::Float64=1.0) where T
    featdims, timesteps, batchsize = size(x)
    loglikely = zeros(eltype(x), batchsize)
    p = softmax(ᵛ(x); dims=1)
    r = zero(ᵛ(x))

    Threads.@threads for b = 1:batchsize
        r[:,:,b], loglikely[b] = TCS(p[:,:,b], seqlabels[b], background=background, foreground=foreground)
    end

    Δ = p - r
    reduce3d(Δ, loglikely, seqlabels, reduction)
    y = Variable{T}([sum(loglikely)], x.backprop)

    if y.backprop
        y.backward = function CRNN_Batch_TCS_With_Softmax_Backward()
            if need2computeδ!(x)
                if weight==1.0
                    δ(x) .+= δ(y) .* Δ
                else
                    δ(x) .+= δ(y) .* Δ .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end
