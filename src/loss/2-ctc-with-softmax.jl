export DNNSoftmaxCTCLossSingleSeq
export FNNSoftmaxCTCLoss
export RNNSoftmaxCTCLoss
export FRNNSoftmaxCTCLoss
export FRNNSoftmaxFocalCTCLoss
export FRNNSoftmaxCTCProbs
export SoftmaxCTCFocalCELoss
export SoftmaxCTCInvPowerCELoss

"""
    DNNSoftmaxCTCLossSingleSeq(x::Variable{T}, seq::VecInt; blank::Int=1)

case batchsize==1 for test case. `x` is the output of a whole complete input sequence

# Inputs
`x`      : 2-D Variable, input sequence\n
`seq`    : 1-D Array, input sequence's label\n

# Structure
    ┌───┐          ┌───┐
    │ │ │ softmax  │ │ │   ┌───────┐
    │ X ├─────────►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘       │
      │              │+        ▼
    ┌─┴─┐            ▼       ┌─┴─┐
    │ │ │          ┌─┴─┐ -   │ │ │
    │ δ │◄─────────┤ - │──◄──┤ r │
    │ │ │          └───┘     │ │ │
    └───┘                    └───┘
"""
function DNNSoftmaxCTCLossSingleSeq(x::Variable{T}, seq::VecInt; blank::Int=1) where T
    p = softmax(ᵛ(x), dims=1)
    L = length(seq) * 2 + 1
    r, nlnp = CTC(p, seq, blank=blank)

    Δ = p - r
    y = Variable{T}([nlnp], x.backprop)

    if y.backprop
        y.backward = function ∇DNNSoftmaxCTCLossSingleSeq()
            if need2computeδ!(x)
                x ← δ(y) .* Δ
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    FNNSoftmaxCTCLoss(x::Variable{T}, seqlabels::VecVecInt, inputlens; blank=1) where T

a batch of concatenated input sequence is processed by neural networks into `x`

# Inputs
`x`         : 2-D Variable, inputs of softmax\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : records each input sequence's length, like [20,17,...]\n

# Structure
    ┌───┐          ┌───┐
    │ │ │ softmax  │ │ │   ┌───────┐
    │ X ├─────────►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘       │
      │              │+        ▼
    ┌─┴─┐            ▼       ┌─┴─┐
    │ │ │          ┌─┴─┐ -   │ │ │
    │ δ │◄─────────┤ - │──◄──┤ r │
    │ │ │          └───┘     │ │ │
    └───┘                    └───┘
"""
function FNNSoftmaxCTCLoss(x::Variable{T},
                           seqlabels::VecVecInt,
                           inputlens::VecInt;
                           blank::Int=1) where T
    batchsize = length(inputLengths)
    nlnp = zeros(eltype(x), batchsize)
    I, F = indexbounds(inputlens)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)

    for b = 1:batchsize
        span = I[b]:F[b]
        r[:,span], nlnp[b] = CTC(p[:,span], seqlabels[b], blank=blank)
    end

    Δ = p - r
    y = Variable{T}([sum(nlnp)], x.backprop)

    if y.backprop
        y.backward = function ∇FNNSoftmaxCTCLoss()
            if need2computeδ!(x)
                x ← δ(y) .* Δ
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    RNNSoftmaxCTCLoss(x::Variable{T}, seqlabels::VecVecInt, inputlens; blank=1) where T

a batch of padded input sequence is processed by neural networks into `x`

# Inputs
`x`         : 3-D Variable with shape (featdims,timesteps,batchsize), inputs of softmax\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`inputlens` : each input's length, like [19,97,...]\n

# Structure
    ┌───┐          ┌───┐
    │ │ │ softmax  │ │ │   ┌───────┐
    │ X ├─────────►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘       │
      │              │+        ▼
    ┌─┴─┐            ▼       ┌─┴─┐
    │ │ │          ┌─┴─┐ -   │ │ │
    │ δ │◄─────────┤ - │──◄──┤ r │
    │ │ │          └───┘     │ │ │
    └───┘                    └───┘
"""
function RNNSoftmaxCTCLoss(x::Variable{T},
                           seqlabels::VecVecInt,
                           inputlens::VecInt;
                           blank::Int=1) where T
    batchsize = length(inputlens)
    nlnp = zeros(eltype(x), 1, 1, batchsize)
    p = zero(ᵛ(x))
    r = zero(p)

    for b = 1:batchsize
        Tᵇ = inputlens[b]
        p[:,1:Tᵇ,b] = softmax(x.value[:,1:Tᵇ,b]; dims=1)
        r[:,1:Tᵇ,b], nlnp[b] = CTC(p[:,1:Tᵇ,b], seqlabels[b], blank=blank)
    end

    Δ = p - r
    l = T(nlnp)
    reduce3d(Δ, l, seqlabels, reduction)
    y = Variable{T}([sum(l)], x.backprop)

    if y.backprop
        y.backward = function ∇RNNSoftmaxCTCLoss()
            if need2computeδ!(x)
                x ← δ(y) .* Δ
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    FRNNSoftmaxCTCLoss(x::Variable{T}, seqlabels::VecVecInt; blank=1) where T

a batch of padded input sequence is processed by neural networks into `x`

# Inputs
`x`         : 3-D Variable (featdims,timesteps,batchsize), inputs of softmax\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n

# Structure
    ┌───┐          ┌───┐
    │ │ │ softmax  │ │ │   ┌───────┐
    │ X ├─────────►│ P ├──►│CTCLOSS│◄── (seqLabel)
    │ │ │          │ │ │   └───┬───┘
    └─┬─┘          └─┬─┘       │
      │              │+        ▼
    ┌─┴─┐            ▼       ┌─┴─┐
    │ │ │          ┌─┴─┐ -   │ │ │
    │ δ │◄─────────┤ - │──◄──┤ r │
    │ │ │          └───┘     │ │ │
    └───┘                    └───┘
"""
function FRNNSoftmaxCTCLoss(x::Variable{T},
                            seqlabels::VecVecInt;
                            reduction::String="seqlen",
                            blank::Int=1) where T
    featdims, timesteps, batchsize = size(x)
    nlnp = zeros(eltype(x), 1, 1, batchsize)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = CTC(p[:,:,b], seqlabels[b], blank=blank)
    end

    Δ = p - r
    l = T(nlnp)
    reduce3d(Δ, l, seqlabels, reduction)
    y = Variable{T}([sum(l)], x.backprop)

    if y.backprop
        y.backward = function ∇FRNNSoftmaxCTCLoss()
            if need2computeδ!(x)
                x ← δ(y) .* Δ
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end



function FRNNSoftmaxFocalCTCLoss(x::Variable{T},
                                 seqlabels::VecVecInt;
                                 reduction::String="seqlen",
                                 focus::Real=1.0f0,
                                 blank::Int=1) where T
    featdims, timesteps, batchsize = size(x)
    S = eltype(x)
    nlnp = zeros(S, 1, 1, batchsize)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)
    𝜸 = S(focus)
    𝟙 = S(1.0f0)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = CTC(p[:,:,b], seqlabels[b], blank=blank)
    end

    𝒍𝒏𝒑 = T(-nlnp)
    𝒑 = exp(𝒍𝒏𝒑)
    𝒌 = @.  (𝟙 - 𝒑)^(𝜸-𝟙) * (𝟙 - 𝒑 - 𝜸*𝒑*𝒍𝒏𝒑)
    t = @. -(𝟙 - 𝒑)^𝜸 * 𝒍𝒏𝒑
    Δ = p - r
    reduce3d(Δ, t, seqlabels, reduction)
    y = Variable{T}([sum(t)], x.backprop)

    if y.backprop
        y.backward = function ∇FRNNSoftmaxFocalCTCLoss()
            if need2computeδ!(x)
                x ← δ(y) .* 𝒌 .* Δ
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    FRNNSoftmaxCTCProbs(x::Variable, seqlabels::VecVecInt; blank::Int=1) -> p::Variable

# Inputs
`x`         : 3-D Variable (featdims,timesteps,batchsize), input of softmax\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n

# Output
`p`         : 3-D Variable (1,1,batchsize), i.e. `p` is the probabilities of each sequence
"""
function FRNNSoftmaxCTCProbs(x::Variable{T}, seqlabels::VecVecInt; blank::Int=1) where T
    S = eltype(x)
    featdims, timesteps, batchsize = size(x)
    nlnp = zeros(S, 1, 1, batchsize)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = CTC(p[:,:,b], seqlabels[b], blank=blank)
    end

    𝒑 = Variable{T}(exp(T(-nlnp)), x.backprop)
    Δ = r - p

    if 𝒑.backprop
        𝒑.backward = function ∇FRNNSoftmaxCTCProbs()
            if need2computeδ!(x)
                x ← δ(𝒑) .* ᵛ(𝒑) .*  Δ
            end
            ifNotKeepδThenFreeδ!(𝒑)
        end
        addchild(𝒑, x)
    end
    return 𝒑
end


function SoftmaxCTCFocalCELoss(x::Variable,
                               seqlabels::VecVecInt;
                               reduction::String="seqlen",
                               focus::Real=0.5f0,
                               blank::Int=1)

    featdims, timesteps, batchsize = size(x)
    p = softmax(x, dims=1)
    r = zero(ᵛ(x))

    for b = 1:batchsize
        r[:,:,b], _ = CTC(p.value[:,:,b], seqlabels[b], blank=blank)
    end
    fce = FocalCE(p, r, focus=focus)
    return Loss(weightseqvar(fce, seqlabels, reduction))
end


function SoftmaxCTCInvPowerCELoss(x::Variable,
                                  seqlabels::VecVecInt;
                                  reduction::String="seqlen",
                                  blank::Int=1,
                                  a::Int=0.3f0,
                                  n::Int=1.0f0)
    featdims, timesteps, batchsize = size(x)
    p = softmax(x, dims=1)
    r = zero(ᵛ(x))

    for b = 1:batchsize
        r[:,:,b], _ = CTC(p.value[:,:,b], seqlabels[b], blank=blank)
    end
    ce = InvPowerCrossEntropy(p, r, a=a, n=n)
    return Loss(weightseqvar(ce, seqlabels, reduction))
end
