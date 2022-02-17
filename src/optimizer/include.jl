abstract type Optimizer end
export Optimizer


# optimizers with L1 and L2 weight decay
include("./1-SGDL1L2.jl")
include("./2-MomentumL1L2.jl")
include("./3-AdamL1L2.jl")
include("./4-AdaGradL1L2.jl")
include("./5-RMSPropL1L2.jl")

# auto gradient clipping
include("./auto-grad-cliper.jl")

# normal lp cliper
include("./cliper.jl")

# learning rates
include("./learn-rate.jl")

# update and zerograds
include("./misc.jl")
