abstract type Optimizer end
export Optimizer


# optimizers with L1 and L2 weight decay
include("./1-SGD.jl")
include("./2-Momentum.jl")
include("./3-Adam.jl")
include("./4-AdaGrad.jl")
include("./5-RMSProp.jl")

# auto gradient clipping
include("./auto-grad-cliper.jl")

# normal lp cliper
include("./cliper.jl")

# learning rates
include("./learn-rate.jl")

# update and zerograds
include("./misc.jl")

include("./regularize.jl")
