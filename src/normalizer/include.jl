abstract type Normalizer <: Block end

export Normalizer

export MeanNorm
include("./MeanNorm.jl")


include("./mean.jl")
include("./znorm.jl")
include("./batchnorm.jl")

export wnorm
include("./weight-normer.jl")
