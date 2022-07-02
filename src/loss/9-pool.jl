export onehotpool
export multihotpool
export PoolLoss

const IntOrNil  = Union{Nothing, Int}
const RealOrNil = Union{Nothing, Real}


"""
    onehotpool(l::VecVecInt,            # label indices e.g. [[2,3],[4,5,5]]
               C::Int,                  # number of categories
               B::Int;                  # batch size
               blank::Int=1,            # blank state's index
               dtype::DataType=Float32) # data type for label variable

    This is designed for classification layer with softmax and returns a bag level label..

# example
    julia> Mira.onehotpool([ [2,3], [4,3,3] ], 4, 2, blank=1) # order and occurrence counts not matter
    4×1×2 Array{Float32, 3}:
    [:, :, 1] =
     1.0
     1.0
     1.0
     0.0

    [:, :, 2] =
     1.0
     0.0
     1.0
     1.0
"""
function onehotpool(l::VecVecInt,
                    C::Int,
                    B::Int;
                    blank::Int=1,
                    dtype::DataType=Float32)
    𝟙 = dtype(1.0f0)
    y = zeros(dtype, C, 1, B)

    for b in 1:B
        for c in l[b]
            if c==0
                break
            end
            y[c,1,b] = 𝟙
        end
        y[blank,1,b] = 𝟙
    end
    return y
end


"""
    multihotpool(l::VecVecInt,              # label indices e.g. [[2,3],[4,5,5]]
                 C::Int,                    # number of categories
                 B::Int;                    # batch size
                 dtype::DataType=Float32)   # data type for label variable

    This is designed for classification layer with sigmoid and returns a bag level label.

# Example
    julia> Mira.multihotpool([ [2,3], [4,3,3] ], 4, 2) # order and occurrence counts not matter
    4×1×2 Array{Float32, 3}:
    [:, :, 1] =
     0.0
     1.0
     1.0
     0.0

    [:, :, 2] =
     0.0
     0.0
     1.0
     1.0
"""
function multihotpool(l::VecVecInt,
                      C::Int,
                      B::Int;
                      dtype::DataType=Float32)
    𝟙 = dtype(1.0f0)
    y = zeros(dtype, C, 1, B)

    for b in 1:B
        for c in l[b]
            if c==0
                break
            end
            y[c,1,b] = 𝟙
        end
    end
    return y
end



"""
    PoolLoss(p::Variable,
             seqlabels::VecVecInt;           # weakly supervised label, e.g. [[3,4,4],[2,2,2]] is the same as [[3,4],[2]]
             reduction::String="sum",        # chose sum or mean
             poolingfn::Function=linearpool, # aggresive function e.g. exppool/powerpool/linearpool
             blank::IntOrNil=nothing,        # when p is the output of softmax then blank is an integer
             focus::RealOrNil=nothing,       # if using focal loss, then focus is the focal param ∈ [0, Inf)
             alpha::Real=0.50000000f0)       # alpha is the weight for positive class, (1-alpha) for negative class.

           frame wise probs                pooled probs
           ┌─────────────┐                  ┌────┐
         ┌─┴───────────┐ │                ┌─┴──┐ │
       ┌─┴───────────┐ │ │              ┌─┴──┐ │ │
    C  │             │ ├─┘    ──►    C  │    │ ├─┘
       │             ├─┘  B             │    ├─┘   B
       └─────────────┘                  └────┘
              T                           1
"""
function PoolLoss(p::Variable{S},
                  seqlabels::VecVecInt;
                  reduction::String="sum",
                  poolingfn::Function=linearpool,
                  blank::IntOrNil=nothing,
                  focus::RealOrNil=nothing,
                  alpha::Real=0.50000000f0) where S

    C, T, B = size(p)
    y = poolingfn(p, dims=2)

    if isnothing(blank)
        # p is the output of sigmoid
        label = multihotpool(seqlabels, C, B, dtype=eltype(p))
    else
        # p is the output of softmax
        label = onehotpool(seqlabels, C, B, blank=blank, dtype=eltype(p))
    end

    if isnothing(focus)
        return BinaryCrossEntropyLoss(y, S(label), reduction=reduction)
    else
        return FocalBCELoss(y, S(label), focus=focus, alpha=alpha, reduction=reduction)
    end
end
