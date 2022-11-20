export SoftmaxCTCLikeLoss
"""
    SoftmaxCTCLikeLoss(x::Variable,
                       seqlabels::VecVecInt;
                       reduction::String="seqlen",
                       gammafn::Function=FastCTC)
# Inputs
`x`         : 3-D Variable, inputs of softmax\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`reduction` : one of seqlen/timesteps/trellis/normal/nil\n
`gammafn`   : like gammafn(x, l) = CTC(x, l, blank=1990), which only acceps 2 \n
arguments, the first is the 2-d probability, 2nd the corresponding sequential label.
"""
function SoftmaxCTCLikeLoss(x::Variable{T},
                            seqlabels::VecVecInt;
                            reduction::String="seqlen",
                            gammafn::Function=FastCTC) where T

    featdims, timesteps, batchsize = size(x)
    nlnp = zeros(eltype(x), 1, 1, batchsize)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = gammafn(p[:,:,b], seqlabels[b])
    end

    Δ = p - r
    l = T(nlnp)
    reduce3d(Δ, l, seqlabels, reduction)
    y = Variable{T}([sum(l)], x.backprop)

    if y.backprop
        y.backward = function ∇SoftmaxCTCLikeLoss()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* Δ
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export SoftmaxFocalCTCLikeLoss
function SoftmaxFocalCTCLikeLoss(x::Variable{T},
                                 seqlabels::VecVecInt;
                                 reduction::String="seqlen",
                                 gammafn::Function=FastCTC,
                                 focus::Real=1.0f0) where T
    featdims, timesteps, batchsize = size(x)
    S = eltype(x)
    nlnp = zeros(S, 1, 1, batchsize)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)
    𝜸 = S(focus)
    𝟙 = S(1.0f0)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = gammafn(p[:,:,b], seqlabels[b])
    end

    𝒍𝒏𝒑 = T(-nlnp)
    𝒑 = exp(𝒍𝒏𝒑)
    𝒌 = @.  (𝟙 - 𝒑)^(𝜸-𝟙) * (𝟙 - 𝒑 - 𝜸*𝒑*𝒍𝒏𝒑)
    t = @. -(𝟙 - 𝒑)^𝜸 * 𝒍𝒏𝒑
    Δ = p - r
    reduce3d(Δ, t, seqlabels, reduction)
    y = Variable{T}([sum(t)], x.backprop)

    if y.backprop
        y.backward = function ∇SoftmaxFocalCTCLikeLoss()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* 𝒌 .* Δ
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


export SoftmaxCTCLikeProbs
"""
    SoftmaxCTCLikeProbs(x::Variable{T}, seqlabels::VecVecInt; gammafn::Function=FastCTC) -> p::Variable

# Inputs
`x`         : 3-D Variable (featdims,timesteps,batchsize), input of softmax\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`gammafn`   : like gammafn(x, l) = CTC(x, l, blank=1), where x is the input of softmax, l is the sequential label

# Output
`p`         : 3-D Variable (1,1,batchsize), i.e. `p` is the probabilities of each sequence
"""
function SoftmaxCTCLikeProbs(x::Variable{T}, seqlabels::VecVecInt; gammafn::Function=FastCTC) where T
    S = eltype(x)
    featdims, timesteps, batchsize = size(x)
    nlnp = zeros(S, 1, 1, batchsize)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = gammafn(p[:,:,b], seqlabels[b])
    end

    𝒑 = Variable{T}(exp(T(-nlnp)), x.backprop)
    Δ = r - p

    if 𝒑.backprop
        𝒑.backward = function ∇SoftmaxCTCLikeProbs()
            if need2computeδ!(x)
                δ(x) .+= δ(𝒑) .* ᵛ(𝒑) .*  Δ
            end
            ifNotKeepδThenFreeδ!(𝒑)
        end
        addchild(𝒑, x)
    end
    return 𝒑
end


export SoftmaxCTCLikeFocalCELoss
function SoftmaxCTCLikeFocalCELoss(x::Variable{T},
                                   seqlabels::VecVecInt;
                                   reduction::String="seqlen",
                                   gammafn::Function=FastCTC,
                                   focus::Real=0.500000000f0) where T

    featdims, timesteps, batchsize = size(x)
    𝒑 = softmax(ᵛ(x), dims=1)
    𝜸 = zero(𝒑)

    for b = 1:batchsize
        𝜸[:,:,b], _ = gammafn(𝒑[:,:,b], seqlabels[b])
    end

    TO = eltype(𝒑)
    ϵ  = TO(1e-38)
    𝒍  = TO(1.0f0)
    𝒏 = TO(focus)

    p⁺  = 𝒑 .+ ϵ    # a little greater
    p⁻  = 𝒑 .- ϵ    # a little smaller
    𝒍𝒏𝒑 = log.(p⁺)  # alias for log(p)
    𝒍𝒔𝒑 = 𝒍 .- p⁻   # alias for 1 - p

    t = @. - 𝜸 * 𝒍𝒔𝒑 ^ 𝒏 * 𝒍𝒏𝒑
    y = Variable{T}(t, x.backprop)

    if y.backprop
        𝒛 = @. 𝜸 * 𝒍𝒔𝒑^(𝒏-𝒍) * (𝒏 * 𝒑 * 𝒍𝒏𝒑 - 𝒍𝒔𝒑)
        y.backward = function ∇SoftmaxCTCLikeFocalCELoss()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* (𝒛 .- 𝒑 .* sum(𝒛, dims=1))
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return Loss(weightseqvar(y, seqlabels, reduction))
end


export SoftmaxCTCLikeWeightedCELoss
function SoftmaxCTCLikeWeightedCELoss(x::Variable,
                                      seqlabels::VecVecInt;
                                      reduction::String="seqlen",
                                      gammafn::Function=CTC,
                                      weightfn::Function=t->(1-t))
    featdims, timesteps, batchsize = size(x)
    p = softmax(x, dims=1)
    r = zero(ᵛ(x))

    for b = 1:batchsize
        r[:,:,b], _ = gammafn(p.value[:,:,b], seqlabels[b])
    end
    wce = weightfn(p) .* CrossEntropy(p, r)
    return Loss(weightseqvar(wce, seqlabels, reduction))
end



export SoftmaxIterativeCTCLikeLoss
function SoftmaxIterativeCTCLikeLoss(x::Variable{T},
                                     seqlabels::VecVecInt;
                                     reduction::String="seqlen",
                                     blank::Int=1,
                                     ratio::Real=0.9) where T
    featdims, timesteps, batchsize = size(x)
    nlnp = zeros(eltype(x), 1, 1, batchsize)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = CTC(p[:,:,b], seqlabels[b], blank=blank)
    end

    l = T(nlnp)
    Δ = p - modifygamma(r, seqlabels, ratio, blank, T)
    reduce3d(Δ, l, seqlabels, reduction)
    y = Variable{T}([sum(l)], x.backprop)

    if y.backprop
        y.backward = function ∇SoftmaxIterativeCTCLikeLoss()
            if need2computeδ!(x)
                δ(x) .+= δ(y) .* Δ
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end
