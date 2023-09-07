"""
    abstract type Block includes basic network struct like:
    1. Dense, MLP, Linear, Affine
    2. RNN IndRNN LSTM IndLSTM GRU IndGRU
    3. RNNs IRNN IndRNNs RIN LSTM IndLSTMs
    4. Conv/Conv1d/Conv2d/Conv3d/Conv4d/Conv5d/Conv1x1
    5. TransConv/TransConv1d/TransConv2d/TransConv3d/TransConv4d/TransConv5d
    6. Pool
"""
abstract type Block end

function (block::Block)(x::Variable, args...; kwargs...)
    return forward(block, x, args...; kwargs...)
end

function (block::Block)(x::Vector{Variable}, args...; kwargs...)
    return forward(block, x, args...; kwargs...)
end


function (block::Block)(x::AbstractArray, args...; kwargs...)
    return predict(block, x, args...; kwargs...)
end

function (block::Block)(x::Vector{AbstractArray}, args...; kwargs...)
    return predict(block, x, args...; kwargs...)
end

function forward(f::Function, x, args...; kwargs...)
    return f(x, args...; kwargs...)
end

function predict(f::Function, x, args...; kwargs...)
    return f(x, args...; kwargs...)
end
