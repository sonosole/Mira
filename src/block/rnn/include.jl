include("./1-IndRNN.jl")
include("./1-RNN.jl")
include("./2-IndLSTM.jl")
include("./2-LSTM.jl")
include("./3-IndGRU.jl")
include("./3-GRU.jl")
include("./batchrnn.jl")

export RNN
export RNNs
export IndRNN
export IndRNNs
export GRU
export GRUs
export IndGRU
export IndGRUs
export LSTM
export LSTMs
export IndLSTM
export IndLSTMs

export resethidden


global RNNLIST = [RNN, RNNs, IndRNN, IndRNNs,
                  LSTM, LSTMs, IndLSTM, IndLSTMs,
                  GRU, GRUs, IndGRU, IndGRUs];
