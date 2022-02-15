export PadSeqPackBatch
export PackedSeqPredict
export PackedSeqForward


"""
    PadSeqPackBatch(inputs::Vector; eps::Real=0.0) -> output
+ `inputs` <: AbstractArray{Real,2}
+ `output` <: AbstractArray{Real,3}
pad epsilon to align raw input features probably with different length
# Examples
    julia> PadSeqPackBatch([ones(2,1), 2ones(2,2), 3ones(2,3)])
    2×3×3 Array{Float64,3}:
    [:, :, 1] =
     1.0  0.0  0.0
     1.0  0.0  0.0

    [:, :, 2] =
     2.0  2.0  0.0
     2.0  2.0  0.0

    [:, :, 3] =
     3.0  3.0  3.0
     3.0  3.0  3.0
"""
function PadSeqPackBatch(inputs::Vector; eps::Real=0.0)
    # all Array of inputs shall have the same size in dim-1
    batchSize = length(inputs)
    lengths   = [size(inputs[i], 2) for i in 1:batchSize]
    featDims  = size(inputs[1], 1)
    maxSteps  = maximum(lengths)
    RNNBatch  = zeros(eltype(inputs[1]), featDims, maxSteps, batchSize)
    fill!(RNNBatch, eps)

    for i = 1:batchSize
        Tᵢ = lengths[i]
        RNNBatch[:,1:Tᵢ,i] .= inputs[i]
    end
    return RNNBatch
end


function PackedSeqForward(chain::Block, x::Variable{S}) where S
    T = size(x, 2)
    v = Vector{Variable{S}}(undef, T)
    resethidden(chain)
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
        y.backward = function PackSeqSlicesBackward()
            Threads.@threads for t = 1:T
                if need2computeδ!(v[t])
                    v[t].delta .+= y.delta[:,t,:]
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, v[T])
    end
    return y
end


function PackedSeqPredict(chain::Block, x::AbstractArray{S}) where S
    T = size(x,2)
    v = Vector{AbstractArray{S}}(undef,T)
    resethidden(chain)
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
