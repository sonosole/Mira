export FunOrNil

const Nil = Nothing
const nil = nothing

const FunOrNil  = Union{Nothing, Function}
const IntOrNil  = Union{Nothing, Int}
const RealOrNil = Union{Nothing, Real}

export VecVecInt
export VecInt
export Dimtype
export IntOrDims

# convinient for ctc-like labels
const    VecInt =        Vector{Int}
const VecVecInt = Vector{Vector{Int}}

# type for kwarg in functions like sum(x,dims=1)
const Dimtype = Union{Int,NTuple{N,Int}} where {N}
const IntOrDims{N} = Union{Int, Dims{N}} where N
const AtArray = AbstractArray
