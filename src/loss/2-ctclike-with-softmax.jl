export SoftmaxCTCLikeLoss
"""
    SoftmaxCTCLikeLoss(x::Variable,
                       seqlabels::VecVecInt;
                       reduction::String="seqlen",
                       pathfn::Function=FastCTC)
# Inputs
+ `x`         : 3-D Variable, inputs of softmax\n
+ `seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
+ `reduction` : one of seqlen/timesteps/trellis/normal/nil\n
+ `pathfn`    : function to calculate soft or hard alignment like :
## pathfn example
    pathfn(x, l) = CTC(x, l, blank=1990)
  which only acceps 2 arguments, the first is the 2-d probability, 2nd the corresponding sequential label.
# Mechanism
    y = softmax(x)
    p = CTCLikeProb(y, seqlabels)
    l = -log(p)
"""
function SoftmaxCTCLikeLoss(x::Variable{T},
                            seqlabels::VecVecInt;
                            reduction::String="seqlen",
                            pathfn::Function=FastCTC) where T

    featdims, timesteps, batchsize = size(x)
    nlnp = zeros(eltype(x), 1, 1, batchsize)
    y = softmax(ᵛ(x), dims=1)
    r = zero(y)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = pathfn(y[:,:,b], seqlabels[b])
    end

    Δ = y - r
    l = T(nlnp)
    reduce3d(Δ, l, seqlabels, reduction)
    c = Variable{T}([sum(l)], x.backprop)

    if c.backprop
        c.backward = function ∇SoftmaxCTCLikeLoss()
            if need2computeδ!(x)
                δ(x) .+= δ(c) .* Δ
            end
            ifNotKeepδThenFreeδ!(c)
        end
        addchild(c, x)
    end
    return c
end


export SoftmaxFocalCTCLikeLoss
"""
    SoftmaxFocalCTCLikeLoss(x::Variable,
                            seqlabels::VecVecInt;
                            reduction::String="seqlen",
                            pathfn::Function=FastCTC,
                            focus::Real=1.0f0)
# Inputs
+ `x`         : 3-D Variable, inputs of softmax\n
+ `seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
+ `reduction` : one of seqlen/timesteps/trellis/normal/nil\n
+ `pathfn`    : function to calculate soft or hard alignment like :
## pathfn example
    pathfn(x, l) = CTC(x, l, blank=1990)
  which only acceps 2 arguments, the first is the 2-d probability, 2nd the corresponding sequential label.
# Mechanism
    y = softmax(x)
    p = CTCLikeProb(y, seqlabels)
    l = (1-p)ᶠ[-log(p)]
where `ᶠ` is the focus param.
"""
function SoftmaxFocalCTCLikeLoss(x::Variable{T},
                                 seqlabels::VecVecInt;
                                 reduction::String="seqlen",
                                 pathfn::Function=FastCTC,
                                 focus::Real=1.0f0) where T
    featdims, timesteps, batchsize = size(x)
    typed = eltype(x)
    nlnp  = zeros(typed, 1, 1, batchsize)
    y = softmax(ᵛ(x), dims=1)
    r = zero(y)         # path cache
    f = typed(focus)    # alias for focus param
    l = typed(1.0f0)    # alias for value 1.0

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = pathfn(y[:,:,b], seqlabels[b])
    end

    lnp = T(-nlnp)
    p   = @. exp(lnp)   # all path's probability
    lsp = @. (l - p)    # alias for value 1 - p

    L = @. lsp^f * (-lnp)   # focal loss
    ζ = @. lsp^(f-l) * (lsp - f * p * lnp)

    Δ = y - r
    reduce3d(Δ, L, seqlabels, reduction)
    c = Variable{T}([sum(L)], x.backprop)

    if c.backprop
        c.backward = function ∇SoftmaxFocalCTCLikeLoss()
            if need2computeδ!(x)
                δ(x) .+= δ(c) .* ζ .* Δ
            end
            ifNotKeepδThenFreeδ!(c)
        end
        addchild(c, x)
    end
    return c
end


export SoftmaxCTCLikeProbs
"""
    SoftmaxCTCLikeProbs(x::Variable{T}, seqlabels::VecVecInt; pathfn::Function=FastCTC) -> p::Variable

# Inputs
`x`         : 3-D Variable (featdims,timesteps,batchsize), input of softmax\n
`seqlabels` : a batch of sequential labels, like [[i,j,k],[x,y],...]\n
`pathfn`   : like pathfn(x, l) = CTC(x, l, blank=1), where x is the input of softmax, l is the sequential label

# Output
`p`         : 3-D Variable (1,1,batchsize), i.e. `p` is the probabilities of each sequence

# Mechanism
    y = softmax(x)
    p = CTCLikeProb(y, seqlabels)
"""
function SoftmaxCTCLikeProbs(x::Variable{T}, seqlabels::VecVecInt; pathfn::Function=FastCTC) where T
    featdims, timesteps, batchsize = size(x)
    nlnp = zeros(eltype(x), 1, 1, batchsize)
    y = softmax(ᵛ(x), dims=1)
    r = zero(p)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = pathfn(y[:,:,b], seqlabels[b])
    end

    p = Variable{T}(exp(T(-nlnp)), x.backprop)

    if p.backprop
        Δ = ᵛ(p) .* (r - y)
        p.backward = function ∇SoftmaxCTCLikeProbs()
            if need2computeδ!(x)
                δ(x) .+= δ(p) .* Δ
            end
            ifNotKeepδThenFreeδ!(p)
        end
        addchild(p, x)
    end
    return p
end


export SoftmaxCTCLikeFocalCELoss
function SoftmaxCTCLikeFocalCELoss(x::Variable{T},
                                   seqlabels::VecVecInt;
                                   reduction::String="seqlen",
                                   pathfn::Function=FastCTC,
                                   focus::Real=0.500000000f0) where T

    featdims, timesteps, batchsize = size(x)
    p = softmax(ᵛ(x), dims=1)
    γ = zero(p)

    for b = 1:batchsize
        γ[:,:,b], _ = pathfn(p[:,:,b], seqlabels[b])
    end

    TO = eltype(p)
    ϵ  = TO(1e-38)
    l  = TO(1.0f0)
    f  = TO(focus)

    p⁺  = p .+ ϵ    # a little greater
    p⁻  = p .- ϵ    # a little smaller
    lnp = log.(p⁺)  # alias for log(p)
    lsp = l .- p⁻   # alias for 1 - p

    c = @. lsp ^ f * (- γ * lnp)
    C = Variable{T}(c, x.backprop)

    if C.backprop
        ṗp = @. γ * lsp^(f-l) * (f * p * lnp - lsp)
        C.backward = function ∇SoftmaxCTCLikeFocalCELoss()
            if need2computeδ!(x)
                ṗp   .*= δ(C)
                δ(x) .+= ṗp .- p .* sum(ṗp, dims=1)
            end
            ifNotKeepδThenFreeδ!(C)
        end
        addchild(C, x)
    end
    return Loss(weightseqvar(C, seqlabels, reduction))
end


export SoftmaxCTCLikeWeightedCELoss
function SoftmaxCTCLikeWeightedCELoss(x::Variable,
                                      seqlabels::VecVecInt;
                                      reduction::String="seqlen",
                                      pathfn::Function=CTC,
                                      weightfn::Function=t->(1-t))
    featdims, timesteps, batchsize = size(x)
    p = softmax(x, dims=1)
    r = zero(ᵛ(x))

    for b = 1:batchsize
        r[:,:,b], _ = pathfn(p.value[:,:,b], seqlabels[b])
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
