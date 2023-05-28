export FastCTC, seqfastctc
export FastCTCGreedySearch
export FastCTCGreedySearchWithTimestamp

export FRNNSoftmaxFastCTCLoss
export FRNNFastCTCLoss
export FRNNSoftmaxFocalFastCTCLoss
export FRNNFocalFastCTCLoss
export ViterbiFastCTC

function seqfastctc(seq::VecInt, blank::Int=1)
    if seq[1] == 0
        return [blank]
    end
    L = length(seq) # sequence length
    N = 2 * L + 1   # topology length
    label = zeros(Int, N)
    label[1:2:N] .= blank
    label[2:2:N] .= seq
    return label
end


"""
    FastCTC(p::Array{T,2}, seqlabel::VecInt; blank::Int=1)

# Topology Example
     ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐
    ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐
    │blank├─►│  C  ├─►│blank├─►│  A  ├─►│blank├─►│  T  ├─►│blank│
    └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
"""
function FastCTC(p::Array{TYPE,2}, seqlabel::VecInt; blank::Int=1) where TYPE
    seq  = seqfastctc(seqlabel, blank)
    ZERO = TYPE(0)                               # typed zero,e.g. Float32(0)
    S, T = size(p)                               # assert p is a 2-D tensor
    L = length(seq)                              # topology length with blanks
    r = fill!(Array{TYPE,2}(undef,S,T), ZERO)    # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

    if L == 1
        r[blank,:] .= TYPE(1)
        return r, - sum(log.(p[blank,:]))
    end

    Log0 = LogZero(TYPE)                         # approximate -Inf of TYPE
    a = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝜶 = p(s[k,t], x[1:t]), k in FastCTC topology's indexing
    b = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝛃 = p(x[t+1:T] | s[k,t]), k in FastCTC topology's indexing
    a[1,1] = log(p[seq[1],1])
    a[2,1] = log(p[seq[2],1])
    b[L-1,T] = ZERO
    b[L  ,T] = ZERO

    # --- forward in log scale ---
    for t = 2:T
        τ = t-1
        first = max(1, t-T+L-1)
        lasst = min(1+t, L)
        for s = first:lasst
            if s≠1
                a[s,t] = LogSum2Exp(a[s,τ], a[s-1,τ])
            else
                a[s,t] = a[s,τ]
            end
            a[s,t] += log(p[seq[s],t])
        end
    end

    # --- backward in log scale ---
    for t = T-1:-1:1
        τ = t+1
        first = max(1, t-T+L-1)
        lasst = min(1+t, L)
        for s = first:lasst
            Q = b[s,τ] + log(p[seq[s],τ])
            if s≠L
                b[s,t] = LogSum2Exp(Q, b[s+1,τ] + log(p[seq[s+1],τ]))
            else
                b[s,t] = Q
            end
        end
    end

    logsum = LogSum2Exp(a[1,1] + b[1,1], a[2,1] + b[2,1])
    g = exp.((a + b) .- logsum)

    # reduce first line of g
    r[blank,:] .+= g[1,:]
    # reduce rest lines of g
    for n = 1:div(L-1,2)
        s = n<<1
        r[seq[s],:] .+= g[s,  :]
        r[blank, :] .+= g[s+1,:]
    end

    return r, -logsum
end


"""
    FastCTCGreedySearch(x::Array; blank::Int=1, dims::Dimtype=1) -> hypothesis
remove repeats and blanks of argmax(x, dims=dims)
"""
function FastCTCGreedySearch(x::Array; blank::Int=1, dims::Dimtype=1)
    hyp = Vector{Int}(undef, 0)
    idx = argmax(x, dims=dims)

    # first time-step
    previous = 0
    current  = idx[1][1]
    if current ≠ blank
        push!(hyp, current)
    end
    # rest time-steps
    for t = 2:length(idx)
        previous = current
        current  = idx[t][1]
        if !(current==previous || current==blank)
            push!(hyp, current)
        end
    end
    return hyp
end


"""
    FastCTCGreedySearchWithTimestamp(x::Array; blank::Int=1, dim::Int=1) -> hypothesis, timestamp
"""
function FastCTCGreedySearchWithTimestamp(x::Array; blank::Int=1, dims::Dimtype=1)
    hyp = Vector{Int}(undef, 0)
    stp = Vector{Float32}(undef, 0)
    idx = argmax(x, dims=dims)
    T   = length(idx)

    # first time-step
    previous = 0
    current  = idx[1][1]
    if current ≠ blank
        push!(hyp, current)
        push!(stp, 1 / T)
    end
    # rest time-steps
    for t = 2:T
        previous = current
        current  = idx[t][1]
        if !(current==previous || current==blank)
            push!(hyp, current)
            push!(stp, t / T)
        end
    end
    return hyp, stp
end



function FRNNSoftmaxFastCTCLoss(x::Variable{T},
                                seqlabels::VecVecInt;
                                reduction::String="seqlen",
                                blank::Int=1,
                                weight=1.0) where T
    featdims, timesteps, batchsize = size(x)
    nlnp = zeros(eltype(x), 1, 1, batchsize)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = FastCTC(p[:,:,b], seqlabels[b], blank=blank)
    end

    Δ = p - r
    l = T(nlnp)
    reduce3d(Δ, l, seqlabels, reduction)
    y = Variable{T}([sum(l)], x.backprop)

    if y.backprop
        y.backward = function ∇FRNNSoftmaxFastCTCLoss()
            if need2computeδ!(x)
                if weight==1.0
                    x ← δ(y) .* Δ
                else
                    x ← δ(y) .* Δ .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function FRNNFastCTCLoss(p::Variable{T},
                         seqlabels::VecVecInt;
                         reduction::String="seqlen",
                         blank::Int=1,
                         weight=1.0) where T

    featdims, timesteps, batchsize = size(p)
    nlnp = zeros(eltype(p), 1, 1, batchsize)
    r = zero(ᵛ(p))

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = FastCTC(p.value[:,:,b], seqlabels[b], blank=blank)
    end

    l = T(nlnp)
    reduce3d(r, l, seqlabels, reduction)
    y = Variable{T}([sum(l)], p.backprop)

    if y.backprop
        y.backward = function ∇FRNNFastCTCLoss()
            if need2computeδ!(p)
                if weight==1.0
                    p ← - δ(y) .* r ./ ᵛ(p)
                else
                    p ← - δ(y) .* r ./ ᵛ(p) .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


function FRNNSoftmaxFocalFastCTCLoss(x::Variable{T},
                                     seqlabels::VecVecInt;
                                     reduction::String="seqlen",
                                     blank::Int=1,
                                     focus::Real=1.0f0,
                                     weight=1.0) where T
    featdims, timesteps, batchsize = size(x)
    S = eltype(x)
    nlnp = zeros(S, 1, 1, batchsize)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)
    𝜸 = S(focus)
    𝟙 = S(1.0f0)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = FastCTC(p[:,:,b], seqlabels[b], blank=blank)
    end

    𝒍𝒏𝒑 = T(-nlnp)
    𝒑 = exp(𝒍𝒏𝒑)
    t = @. -(𝟙 - 𝒑)^𝜸 * 𝒍𝒏𝒑   # focal version of CTC loss
    𝒌 = @.  (𝟙 - 𝒑)^(𝜸-𝟙) * (𝟙 - 𝒑 - 𝜸*𝒑*𝒍𝒏𝒑) # 𝒑 * ∂(-t)/∂𝒑

    Δ = p - r
    reduce3d(Δ, t, seqlabels, reduction)
    y = Variable{T}([sum(t)], x.backprop)

    if y.backprop
        y.backward = function ∇FRNNSoftmaxFocalFastCTCLoss()
            if need2computeδ!(x)
                if weight==1.0
                    x ← δ(y) .* 𝒌 .* Δ
                else
                    x ← δ(y) .* 𝒌 .* Δ .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function FRNNFocalFastCTCLoss(p::Variable{T},
                              seqlabels::VecVecInt;
                              reduction::String="seqlen",
                              blank::Int=1,
                              focus::Real=1.0f0,
                              weight=1.0) where T
    featdims, timesteps, batchsize = size(p)
    S = eltype(p)
    nlnp = zeros(S, 1, 1, batchsize)
    r = zero(ᵛ(p))
    𝜸 = S(focus)
    𝟙 = S(1.0f0)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = FastCTC(p.value[:,:,b], seqlabels[b], blank=blank)
    end

    𝒍𝒏𝒑 = T(-nlnp)
    𝒑 = exp(𝒍𝒏𝒑)
    t = @. -(𝟙 - 𝒑)^𝜸 * 𝒍𝒏𝒑       # focal version loss
    𝒌 = @.  (𝟙 - 𝒑)^(𝜸-𝟙) * (𝜸*𝒑*𝒍𝒏𝒑 + 𝒑 - 𝟙) # 𝒑 * ∂t/∂𝒑

    reduce3d(r, t, seqlabels, reduction)
    y = Variable{T}([sum(t)], p.backprop)

    if y.backprop
        y.backward = function ∇FRNNFocalFastCTCLoss()
            if need2computeδ!(p)
                if weight==1.0
                    p ← δ(y) .* 𝒌 .* r ./ ᵛ(p)
                else
                    p ← δ(y) .* 𝒌 .* r ./ ᵛ(p) .* weight
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


function FRNNFastCTCProbs(p::Variable{T}, seqlabels::VecVecInt; blank::Int=1) where T
    featdims, timesteps, batchsize = size(p)
    nlnp = zeros(eltype(p), 1, 1, batchsize)
    r = zero(ᵛ(p))

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = FastCTC(p.value[:,:,b], seqlabels[b], blank=blank)
    end

    𝒑 = Variable{T}(exp(T(-nlnp)), x.backprop)

    if 𝒑.backprop
        𝒑.backward = function ∇FRNNFastCTCProbs()
            if need2computeδ!(p)
                p ← - δ(𝒑) .* r ./ ᵛ(p)
            end
            ifNotKeepδThenFreeδ!(𝒑)
        end
        addchild(𝒑, p)
    end
    return 𝒑
end


function FRNNSoftmaxFastCTCProbs(x::Variable{T}, seqlabels::VecVecInt; blank::Int=1) where T
    featdims, timesteps, batchsize = size(x)
    nlnp = zeros(eltype(x), 1, 1, batchsize)
    p = softmax(ᵛ(x), dims=1)
    r = zero(p)

    for b = 1:batchsize
        r[:,:,b], nlnp[b] = FastCTC(p[:,:,b], seqlabels[b], blank=blank)
    end

    𝒑 = Variable{T}(exp(T(-nlnp)), x.backprop)
    Δ = p - r

    if 𝒑.backprop
        𝒑.backward = function ∇FRNNSoftmaxFastCTCProbs()
            if need2computeδ!(x)
                x ← δ(𝒑) .* Δ
            end
            ifNotKeepδThenFreeδ!(𝒑)
        end
        addchild(𝒑, x)
    end
    return 𝒑
end


"""
    ViterbiFastCTC(p::Array{F,2}, seqlabel::VecInt; blank::Int=1)
force alignment by viterbi algorithm

# Topology Example
     ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐
    ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐
    │blank├─►│  S  ├─►│blank├─►│  U  ├─►│blank├─►│  N  ├─►│blank│
    └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
"""
function ViterbiFastCTC(p::Array{TYPE,2}, seqlabel::VecInt; blank::Int=1, eps::Real=1f-5) where TYPE
    S, T = size(p)                               # assert p is a 2-D tensor
    seq  = seqfastctc(seqlabel, blank)           # extend by topology constraint
    Log0 = LogZero(TYPE)                         # approximate -Inf of TYPE
    ZERO = TYPE(eps / (S-1))                     # typed value closing to 0 but bigger than 0
    ONE  = TYPE(1.0f0 - eps)                     # typed value closing to 1 but less than 1
    lnp  = TYPE(0.0f0)
    L = length(seq)                              # topology length with blanks, assert L ≤ T
    r = fill!(Array{TYPE,2}(undef,S,T), ZERO)    # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

    if L == 1
        r[blank,:] .= ONE
        return r, - sum(log.(p[blank,:]))
    end

    d = fill!(Array{TYPE,2}(undef,L,T), Log0)
    ϕ = zeros(Int, L, T-1)
    h = zeros(Int, T)
    ϵ = TYPE(1e-38)

    # ══ init at fisrt timestep ══
    d[1,1] = log(p[seq[1],1] + ϵ)
    d[2,1] = log(p[seq[2],1] + ϵ)

    # ══ viterbi in log scale ══
    for t = 2:T
        τ = t-1
        first = max(1, t-T+L-1)
        lasst = min(1+t, L)
        if first ≠ 1 # then each node has two kids
            for s = first:lasst
                i = ifelse(d[s-1,τ] > d[s,τ], s-1, s)
                d[s,t] = d[i,τ] + log(p[seq[s],t] + ϵ)
                ϕ[s,τ] = i
            end
        else
            d[first,t] = d[first,τ] + log(p[blank,t] + ϵ)
            ϕ[first,τ] = 1
            for s = first+1:lasst
                i = ifelse(d[s-1,τ] > d[s,τ], s-1, s)
                d[s,t] = d[i,τ] + log(p[seq[s],t] + ϵ)
                ϕ[s,τ] = i
            end
        end
    end
    # ══ backtrace ══
    h[T] = ifelse(d[L,T] > d[L-1,T], L, L-1)
    for t = T-1:-1:1
        h[t] = ϕ[h[t+1],t]
    end
    # ══ one-hot assignment ══
    for t = 1:T
        i = seq[h[t]]
        r[i,t] = ONE
        lnp += log(p[i,t] + ϵ)
    end
    return r, -lnp
end
