module Mira
Base.__precompile__(true)


using Statistics: mean, std
import Statistics.mean

export predict
export forward
export backward
export update!
export zerograds!

include("./kit/include.jl")
include("./base/include.jl")
include("./block/include.jl")
include("./scaler/include.jl")
include("./normalizer/include.jl")
include("./optimizer/include.jl")
include("./quant/include.jl")
include("./loss/include.jl")
include("./engine/include.jl")
include("./metrics/include.jl")



end
