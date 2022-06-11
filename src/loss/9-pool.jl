export onehotpool
export multihotpool
export OnehotLinearPoolLoss
export MultihotLinearPoolLoss
export PoolLoss

const IntOrNil  = Union{Nothing, Int}
const BoolOrNil = Union{Nothing, Bool}


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
                  focus::BoolOrNil=nothing,
                  gamma::Real=1.0) where S

    C, T, B = size(p)
    y = poolingfn(p, dims=2)

    if isnothing(blank) # prob was made by sigmoid
        label = S( multihotpool(seqlabels, C, B, dtype=eltype(p)) )
        if isnothing(focus)
            return binaryCrossEntropyLoss(y, label, reduction=reduction)
        else
            return focalBCELoss(y, label, gamma=gamma, reduction=reduction)
        end
    else # prob was made by softmax
        label = S( onehotpool(seqlabels, C, B, blank=blank, dtype=eltype(p)) )
        if isnothing(focus)
            return crossEntropyLoss(y, label, reduction=reduction)
        else
            return focalCELoss(y, label, gamma=gamma, reduction=reduction)
        end
    end
end
