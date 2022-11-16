export CTC
export CTCGreedySearch
export CTCGreedySearchWithTimestamp
export indexbounds

"""
    indexbounds(lengthArray)
`lengthArray` records length of each sequence, i.e. labels or features
# example
    julia> indexbounds([2,0,3,2])
    ([1; 3; 3; 6], [2; 2; 5; 7])
"""
function indexbounds(lengthArray)
    acc = 0
    num = length(lengthArray)
    s = ones(Int,num,1)
    e = ones(Int,num,1)
    for i = 1:num
        s[i] += acc
        e[i] = s[i] + lengthArray[i] - 1
        acc += lengthArray[i]
    end
    return (s,e)
end


"""
    CTC(p::Array{T,2}, seq::Vector{Int}; blank=1) -> (target, lossvalue)
# Inputs
    p   : probability of softmax output\n
    seq : label seq like [9,3,6,15] which contains no blank. If p
          has no label (e.g. pure noise) then seq is [0]
# Outputs
    target    : target of softmax's output\n
    lossvalue : negative log-likelyhood
"""
function CTC(p::Array{TYPE,2}, seq::VecInt; blank::Int=1) where TYPE
    ZERO = TYPE(0)                               # typed zero,e.g. Float32(0)
    S, T = size(p)                               # assert p is a 2-D tensor
    r = fill!(Array{TYPE,2}(undef,S,T), ZERO)    # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

    if seq[1] == 0                               # no sequence label
        r[blank,:] .= TYPE(1)
        return r, - sum(log.(p[blank,:]))
    end

    Log0 = LogZero(TYPE)                         # approximate -Inf of TYPE
    L = length(seq)*2 + 1                        # topology length with blanks
    a = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝜶 = p(s[k,t], x[1:t]), k in CTC topology's indexing
    b = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝛃 = p(x[t+1:T] | s[k,t]), k in CTC topology's indexing
    a[1,1] = log(p[blank, 1])
    a[2,1] = log(p[seq[1],1])
    b[L-1,T] = ZERO
    b[L-0,T] = ZERO

    # --- forward in log scale ---
    for t = 2:T
        first = max(1,L-2*(T-t)-1);
        lasst = min(2*t,L);
        for s = first:lasst
            i = div(s,2);
            if s==1
                a[s,t] = a[s,t-1] + log(p[blank,t])
            elseif mod(s,2)==1
                a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1]) + log(p[blank,t])
            elseif s==2
                a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1]) + log(p[seq[i],t])
            elseif seq[i]==seq[i-1]
                a[s,t] = LogSum2Exp(a[s,t-1], a[s-1,t-1]) + log(p[seq[i],t])
            else
                a[s,t] = LogSum3Exp(a[s,t-1], a[s-1,t-1], a[s-2,t-1]) + log(p[seq[i],t])
            end
        end
    end

    # --- backward in log scale ---
    for t = T-1:-1:1
        first = max(1,L-2*(T-t)-1)
        lasst = min(2*t,L)
        for s = first:lasst
            i = div(s,2)
            j = div(s+1,2)
            if s==L
                b[s,t] = b[s,t+1] + log(p[blank,t+1])
            elseif mod(s,2)==1
                b[s,t] = LogSum2Exp(b[s,t+1] + log(p[blank,t+1]), b[s+1,t+1] + log(p[seq[j],t+1]))
            elseif s==L-1
                b[s,t] = LogSum2Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[blank,t+1]))
            elseif seq[i]==seq[i+1]
                b[s,t] = LogSum2Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[blank,t+1]))
            else
                b[s,t] = LogSum3Exp(b[s,t+1] + log(p[seq[i],t+1]), b[s+1,t+1] + log(p[blank,t+1]), b[s+2,t+1] + log(p[seq[i+1],t+1]))
            end
        end
    end

    logsum = LogSum3Exp(Log0, a[1,1] + b[1,1], a[2,1] + b[2,1])
    g = exp.((a + b) .- logsum)

    # reduce first line of g
    r[blank,:] .+= g[1,:]
    # reduce rest lines of g
    for n = 1:length(seq)
        s = n<<1
        r[seq[n],:] .+= g[s,  :]
        r[blank, :] .+= g[s+1,:]
    end

    return r, -logsum
end


"""
    CTCGreedySearch(x::Array; blank=1, dims=1) -> hypothesis
remove repeats and blanks of argmax(x, dims=dims)
"""
function CTCGreedySearch(x::Array; blank::Int=1, dims::Dimtype=1)
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
    CTCGreedySearchWithTimestamp(x::Array; blank::Int=1, dims=1) -> hypothesis, timestamp
"""
function CTCGreedySearchWithTimestamp(x::Array; blank::Int=1, dims::Dimtype=1)
    hyp = Vector{Int}(undef, 0)
    stp = Vector{Float32}(undef, 0)
    idx = argmax(x, dims=dims)
    T   = length(idx)

    # first time-step
    previous = 0
    current  = idx[1][1]
    if current ≠ blank
        push!(hyp, current)
        push!(stp, t / T)
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


# Paper: Reinterpreting CTC Training as Iterative Fitting
export CTCLabelRatio
"""
    julia> CTCLabelRatio([ [0] , [2,2,3] ], 4, 2, a=0.4)
    4×1×2 Array{Float32, 3}:
    [:, :, 1] =
    0.6
    0.0
    0.0
    0.0

    [:, :, 2] =
    2.4
    0.8
    0.4
    0.0
"""
function CTCLabelRatio(l::VecVecInt,
                       C::Int,
                       B::Int;
                       a::Real=0.9,
                       blank::Int=1,
                       dtype::DataType=Float32)
    α = dtype(a) # 0 < α < 1
    𝟙 = dtype(1)
    β = 𝟙 - α
    y = zeros(dtype, C, 1, B)

    for b in 1:B
        if l[b][1] ≠ 0
            for c in l[b]
                y[c,1,b] += α
            end
            y[blank,1,b] = β * (𝟙 + length(l[b]))
        else
            y[blank,1,b] = β
        end
    end
    return y
end


export modifygamma
function modifygamma(r::AbstractArray, seqlabels::VecVecInt, a::Real, blank::Int, T::Type)
    C = size(r, 1)          # channels
    B = length(seqlabels)   # batchsize
    N = T(CTCLabelRatio(seqlabels, C, B, a=a, blank=blank))
    V = sum(r, dims=2)
    γ = r .* (N ./ V)
    return γ ./ sum(γ, dims=1)
end


# Paper: Reinterpreting CTC Training as Iterative Fitting
function weightedgamma(r::AbstractArray, l::VecVecInt, a::Real, blank::Int, T::Type)
    dtype = eltype(T)
    C = size(r, 1)  # channels
    B = length(l)   # batchsize
    α = dtype(a)    # 0 < α < 1
    𝟙 = dtype(1)
    β = 𝟙 - α
    N = zeros(dtype, C, 1, B)

    for b in 1:B
        if l[b][1] ≠ 0
            for c in l[b]
                N[c,1,b] += α
            end
            N[blank,1,b] = β * (𝟙 + length(l[b]))
        else
            N[blank,1,b] = β
        end
    end

    V = sum(r, dims=2)
    γ = r .* (T(N) ./ V)
    return γ ./ sum(γ, dims=1)
end


export CTCLabelFreq
"""
    CTCLabelFreq(l::VecVecInt,
                 C::Int,
                 B::Int;
                 blank::Int=1,
                 dtype::DataType=Float32)

Count the number of occurrences of non-blank and blank states from CTC topology.
Sequence like `[2, 2, 3, 2]` would be converted to `[*, 2, *, 2, *, 3, *, 2, *]`,
where `*` is the blank index.
# Example
    julia> CTCLabelFreq([ [0] , [2,2,3,2] ], 4, 2, blank=1)
    4×1×2 Array{Float32, 3}:
    [:, :, 1] =
     1.0
     0.0
     0.0
     0.0

    [:, :, 2] =
     5.0
     3.0
     1.0
     0.0
"""
function CTCLabelFreq(l::VecVecInt,
                      C::Int,
                      B::Int;
                      blank::Int=1,
                      dtype::DataType=Float32)
    𝟙 = dtype(1)
    y = zeros(dtype, C, 1, B)

    for b in 1:B
        if l[b][1] ≠ 0
            for c in l[b]
                y[c,1,b] += 𝟙
            end
            y[blank,1,b] = 𝟙 + length(l[b])
        else
            y[blank,1,b] = 𝟙
        end
    end
    return y
end


export CTCLabelInvFreq
"""
    CTCLabelInvFreq(l::VecVecInt, # batched sequences
                    C::Int,       # channels
                    B::Int;       # batch size
                    blank::Int=1, # blank index
                    dtype::DataType=Float32)

Count the number of occurrences of non-blank and blank states from CTC topology.
And then take its inverse as weight for those states. Sequence like `[2, 2, 3, 2]`
would be converted to `[*, 2, *, 2, *, 3, *, 2, *]`, where `*` is the blank index.

# Example
    julia> CTCLabelInvFreq([ [0] , [2,2,3,2] ], 4, 2, blank=1)
    4×1×2 Array{Float32, 3}:
    [:, :, 1] =
     1.0
     0.0
     0.0
     0.0

    [:, :, 2] =
     0.2
     0.33333334
     1.0
     0.0
"""
function CTCLabelInvFreq(l::VecVecInt,
                         C::Int,
                         B::Int;
                         blank::Int=1,
                         dtype::DataType=Float32)
    𝟙 = dtype(1)
    y = zeros(dtype, C, 1, B)

    for b in 1:B
        if l[b][1] ≠ 0
            for c in l[b]
                y[c,1,b] += 𝟙
            end
            y[blank,1,b] = 𝟙 / (𝟙 + length(l[b]))
        else
            y[blank,1,b] = 𝟙
        end
    end

    for b in 1:B
        if l[b][1] ≠ 0
            for c in unique(l[b])
                y[c,1,b] = 𝟙 / y[c,1,b]
            end
        end
    end
    return y
end


export CTCLabelWInvFreq
"""
    CTCLabelWInvFreq(l::VecVecInt, # batched sequences
                     C::Int,       # channels
                     B::Int;       # batch size
                     a::Real=0.9,  # weight for non-blank classes
                     blank::Int=1, # blank index
                     dtype::DataType=Float32)

Count the number of occurrences of non-blank and blank states from CTC topology.
And then return its weighted inverse. Sequence like `[2, 2, 3, 2]` would be converted
to `[*, 2, *, 2, *, 3, *, 2, *]`, where `*` is the blank index.

# Example
    julia> CTCLabelWInvFreq([ [0] , [2,2,3,2] ], 4, 2, blank=1)
    4×1×2 Array{Float32, 3}:
    [:, :, 1] =
     1.0
     0.0
     0.0
     0.0

    [:, :, 2] =
     0.2
     0.33333334
     1.0
     0.0
"""
function CTCLabelWInvFreq(l::VecVecInt,
                          C::Int,
                          B::Int;
                          a::Real=0.9,
                          blank::Int=1,
                          dtype::DataType=Float32)
    𝟙 = dtype(1)
    y = zeros(dtype, C, 1, B)
    α = dtype(a)
    β = 𝟙 - α

    for b in 1:B
        if l[b][1] ≠ 0
            for c in l[b]
                y[c,1,b] += 𝟙
            end
            y[blank,1,b] = β / (𝟙 + length(l[b]))
        else
            y[blank,1,b] = β
        end
    end

    for b in 1:B
        if l[b][1] ≠ 0
            for c in unique(l[b])
                y[c,1,b] = α / y[c,1,b]
            end
        end
    end
    return y
end
