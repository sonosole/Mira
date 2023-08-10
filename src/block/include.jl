"""
    abstract type Block includes basic network struct like:
    1. Dense, MLP
    2. RNN irnn IndRNN rin lstm IndLSTM
    3. RNNs IRNN IndRNNs RIN LSTM IndLSTMs
    4. Conv1d/Conv2d/Conv3d/Conv4d/Conv5d

"""
abstract type Block end
export Block
export bytesof, kbytesof, mbytesof, gbytesof, tbytesof
export gradsof
export paramsof
export xparamsof
export nparamsof
export weightsof
export unbiasedof
export nops
export checkvalues
export staticsof

include("./1-chain.jl")
include("./2-residual.jl")
include("./3-dropout.jl")
include("./4-macro.jl")
include("./5-misc.jl")

include("./conv/include.jl")
include("./fc/include.jl")
include("./rnn/include.jl")
include("./pfun/include.jl")



function forward(f::Function, x::Variable, args...; kwargs...)
    return f(x, args...; kwargs...)
end


function predict(f::Function, x::AbstractArray, args...; kwargs...)
    return f(x, args...; kwargs...)
end

# forward(f::Function, x::Variable)      = f(x)
# predict(f::Function, x::AbstractArray) = f(x)
