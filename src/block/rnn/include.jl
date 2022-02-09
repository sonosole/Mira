include("./1-indrnn.jl")
include("./2-indlstm.jl")
include("./3-rnn.jl")
include("./4-batchrnn.jl")

export RNN
export RNNs
export IndRNN
export IndRNNs
export IndLSTM
export IndLSTMs

export resethidden


global RNNLIST = [IndRNN,IndRNNs,
                  IndLSTM,IndLSTMs,
                  RNN,RNNs];
# global RNNLIST = [RNN, rin, lstm, IndRNN, IndLSTM, RNNs, RIN, LSTM, IndLSTMs];
