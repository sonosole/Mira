"""
    abstract type Block includes basic network struct like:
    1. Dense, MLP
    2. RNN irnn IndRNN rin lstm IndLSTM
    3. RNNs IRNN IndRNNs RIN LSTM IndLSTMs
    4. PlainConv1d

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

include("./1-chain.jl")
include("./2-residual.jl")
include("./3-dropout.jl")
include("./4-macro.jl")

include("./conv/include.jl")
include("./fc/include.jl")
include("./rnn/include.jl")
