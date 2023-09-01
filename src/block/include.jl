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

include("./0-abstract.jl")
include("./1-chain.jl")
include("./2-residual.jl")
include("./3-dropout.jl")
include("./4-macro.jl")
include("./5-misc.jl")

include("./conv/include.jl")
include("./fc/include.jl")
include("./rnn/include.jl")
include("./pfun/include.jl")
