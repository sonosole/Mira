export acelabel
export FocalACELoss
export SoftmaxFocalACELoss
export ACELoss
export SoftmaxACELoss


"""
    acelabel(l::VecVecInt,
             C::Int,
             T::Int,
             B::Int;
             blank::Int=1,
             dtype::DataType=Float32)
# Arguments
`l` : batched label sequence, like [[2,2,3],[3,4,5],[0]], in which 0 means nothing\n
`C` : number of output channels (1 + #categories)\n
`T` : timesteps of batched sequence\n
`B` : batchsize\n
`blank` : the same as CTC loss
"""
function acelabel(l::VecVecInt,
                  C::Int,
                  T::Int,
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
            y[c,1,b] += 𝟙
        end
        y[blank,1,b] = T - length(l[b])
    end
    return y ./ T
end


function SoftmaxACELoss(x::Variable{S},
                        seqlabels::VecVecInt;
                        reduction::String="sum",
                        blank::Int=1) where S
    C, T, B = size(x)
    p = softmax(x, dims=1)
    y = mean(p, dims=2)
    label = acelabel(seqlabels, C, T, B, blank=blank, dtype=eltype(x))
    return CrossEntropyLoss(y, S(label), reduction=reduction)
end


function SoftmaxFocalACELoss(x::Variable{S},
                             seqlabels::VecVecInt;
                             reduction::String="sum",
                             gamma::Real=1.0,
                             blank::Int=1) where S
    C, T, B = size(x)
    p = softmax(x, dims=1)
    y = mean(p, dims=2)
    label = acelabel(seqlabels, C, T, B, blank=blank, dtype=eltype(x))
    return FocalCELoss(y, S(label), gamma=gamma, reduction=reduction)
end


"""
           frame wise probs                ace probs
           ┌─────────────┐                  ┌────┐
         ┌─┴───────────┐ │                ┌─┴──┐ │
       ┌─┴───────────┐ │ │              ┌─┴──┐ │ │
    C  │             │ ├─┘    ──►    C  │    │ ├─┘
       │             ├─┘  B             │    ├─┘   B
       └─────────────┘                  └────┘
              T                           1
"""
function ACELoss(p::Variable{S},
                 seqlabels::VecVecInt;
                 reduction::String="sum",
                 blank::Int=1) where S
    C, T, B = size(p)
    y = mean(p, dims=2)
    label = acelabel(seqlabels, C, T, B, blank=blank, dtype=eltype(p))
    return CrossEntropyLoss(y, S(label), reduction=reduction)
end


"""
           frame wise probs                ace probs
           ┌─────────────┐                  ┌────┐
         ┌─┴───────────┐ │                ┌─┴──┐ │
       ┌─┴───────────┐ │ │              ┌─┴──┐ │ │
    C  │             │ ├─┘    ──►    C  │    │ ├─┘
       │             ├─┘  B             │    ├─┘   B
       └─────────────┘                  └────┘
              T                           1
"""
function FocalACELoss(p::Variable{S},
                      seqlabels::VecVecInt;
                      reduction::String="sum",
                      gamma::Real=1.0,
                      blank::Int=1) where S
    C, T, B = size(p)
    y = mean(p, dims=2)
    label = acelabel(seqlabels, C, T, B, blank=blank, dtype=eltype(p))
    return FocalCELoss(y, S(label), gamma=gamma, reduction=reduction)
end
