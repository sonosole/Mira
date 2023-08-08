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

const    VecInt =        Vector{Int}
const VecVecInt = Vector{Vector{Int}}
const Dimtype = Union{Int,NTuple{N,Int}} where {N}
const IntOrDims{N} = Union{Int, Dims{N}} where N
