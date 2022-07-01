export onehotpool
export multihotpool
export PoolLoss

const IntOrNil  = Union{Nothing, Int}
const RealOrNil = Union{Nothing, Real}


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
                  focus::RealOrNil=nothing) where S

    C, T, B = size(p)
    y = poolingfn(p, dims=2)

    if isnothing(blank)
        # prob was made by sigmoid
        label = multihotpool(seqlabels, C, B, dtype=eltype(p))
    else
        # prob was made by softmax
        label = onehotpool(seqlabels, C, B, blank=blank, dtype=eltype(p))
    end

    if isnothing(focus)
        return BinaryCrossEntropyLoss(y, S(label), reduction=reduction)
    else
        return FocalBCELoss(y, S(label), focus=focus, reduction=reduction)
    end
end
