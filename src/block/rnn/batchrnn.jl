export packsequences
export seqpredict
export seqforward


"""
    packsequences(inputs::Vector; eps::Real=0.0, align="random") -> output
Pad sequences of variable length to the same length `T`, where `T` is the length
of the longest sequence. Then pack sequences into a batched `output` for RNN block.
- `inputs` <: Vector{AbstractArray{Real,2}}
- `eps` is the padding value, default to zero
- `align` is the alignment method of all sequence, optionals are:
  - `"random"`, which is the default mode. Sequences are aligned randomly
  - `"left"`, sequences are aligned to the left
  - `"right"`, sequences are aligned to the right
  - `"center"`, sequences are aligned to the center
- `output` <: AbstractArray{Real,3}

# Examples
```julia
julia> packsequences([ones(Int,2,3), 2ones(Int,2,2), 3ones(Int,2,10)], align="random")
2×10×3 Array{Int64, 3}:
[:, :, 1] =
 0  0  0  1  1  1  0  0  0  0
 0  0  0  1  1  1  0  0  0  0
[:, :, 2] =
 0  0  0  2  2  0  0  0  0  0
 0  0  0  2  2  0  0  0  0  0
[:, :, 3] =
 3  3  3  3  3  3  3  3  3  3
 3  3  3  3  3  3  3  3  3  3

julia> packsequences([ones(Int,2,3), 2ones(Int,2,2), 3ones(Int,2,10)], align="center")
2×10×3 Array{Int64, 3}:
[:, :, 1] =
 0  0  0  0  1  1  1  0  0  0
 0  0  0  0  1  1  1  0  0  0
[:, :, 2] =
 0  0  0  0  2  2  0  0  0  0
 0  0  0  0  2  2  0  0  0  0
[:, :, 3] =
 3  3  3  3  3  3  3  3  3  3
 3  3  3  3  3  3  3  3  3  3

julia> packsequences([ones(Int,2,3), 2ones(Int,2,2), 3ones(Int,2,10)], align="right")
2×10×3 Array{Int64, 3}:
[:, :, 1] =
 0  0  0  0  0  0  0  1  1  1
 0  0  0  0  0  0  0  1  1  1
[:, :, 2] =
 0  0  0  0  0  0  0  0  2  2
 0  0  0  0  0  0  0  0  2  2
[:, :, 3] =
 3  3  3  3  3  3  3  3  3  3
 3  3  3  3  3  3  3  3  3  3
```
"""
function packsequences(inputs::Vector; eps::Real=0.0, align="random")
    # all Array of inputs shall have the same size in dim-1
    batchsize = length(inputs)
    lengths   = [size(inputs[i], 2) for i in 1:batchsize]
    featdim   = size(inputs[1], 1)
    maxlen    = maximum(lengths)
    RNNBatch  = zeros(eltype(inputs[1]), featdim, maxlen, batchsize)
    fill!(RNNBatch, eps)
    if isequal(align, "random")
        Threads.@threads for i = 1:batchsize
            T = lengths[i]
            S = rand(1:(maxlen-T+1))
            E = S + T - 1
            RNNBatch[:,S:E,i] .= inputs[i]
        end
    elseif isequal(align, "left")
        Threads.@threads for i = 1:batchsize
            RNNBatch[:,1:lengths[i],i] .= inputs[i]
        end
    elseif isequal(align, "right")
        Threads.@threads for i = 1:batchsize
            S = maxlen - lengths[i] + 1
            RNNBatch[:,S:maxlen,i] .= inputs[i]
        end
    elseif isequal(align, "center")
        Threads.@threads for i = 1:batchsize
            paddings  = maxlen - lengths[i]
            leftpads  = div(paddings, 2, RoundUp)
            rightpads = paddings - leftpads
            S = leftpads + 1
            E = maxlen - rightpads
            RNNBatch[:,S:E,i] .= inputs[i]
        end
    else
        error("align mode $align is not known yet...")
    end
    return RNNBatch
end


function seqforward(chain::Block, x::Variable{S}; keepstate=false) where S
    T = size(x, 2)
    v = Vector{Variable{S}}(undef, T)

    if !keepstate
        if typeof(chain) in RNNSet
            resethidden(chain)
        end
    end

    for t = 1:T
        v[t] = forward(chain, x[:,t,:])
    end

    timeSteps = T
    featsDims = size(v[1], 1)
    batchSize = size(v[1], 2)
    y = Variable{S}(S(undef, featsDims, timeSteps, batchSize), x.backprop)

    Threads.@threads for t = 1:timeSteps
        y.value[:,t,:] .= v[t].value
    end

    if y.backprop
        y.backward = function ∇PackSeqSlices()
            Threads.@threads for t = 1:T
                if needgrad(v[t])
                    v[t] ← y.delta[:,t,:]
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        for t = 1:T
            addchild(y, v[t])
        end
    end
    return y
end


function seqpredict(chain::Block, x::AbstractArray{S}; keepstate=false) where S
    T = size(x,2)
    v = Vector{AbstractArray{S}}(undef,T)

    if !keepstate
        if typeof(chain) in RNNSet
            resethidden(chain)
        end
    end

    for t = 1:T
        v[t] = predict(chain, x[:,t,:])
    end

    timeSteps = T
    featsDims = size(v[1], 1)
    batchSize = size(v[1], 2)
    y = typeof(x)(undef, featsDims, timeSteps, batchSize)
    for t = 1:timeSteps
        y[:,t,:] .= v[t]
    end
    return y  # a rnn batch
end
