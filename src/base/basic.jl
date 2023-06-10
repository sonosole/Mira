export Zeros
export Ones
export vartype

"""
    Zeros(::Type{T}, shape...) where T
return an all-zero-elements-array of type T which has shape `shape...`

# Example
    julia> Zeros(Array{Float64}, 2, 5)
    2×5 Array{Float64,2}:
     0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0
 """
function Zeros(::Type{T}, shape...) where T
    return fill!(T(undef, shape...), 0.0)
end


"""
    Ones(::Type{T}, shape...) where T
return an all-one-elements-array of type T which has shape `shape...`

# Example
    julia> 7Ones(Array{Float64}, 2, 5)
    2×5 Array{Float64,2}:
     7.0  7.0  7.0  7.0  7.0
     7.0  7.0  7.0  7.0  7.0
 """
function Ones(::Type{T}, shape...) where T
    return fill!(T(undef, shape...), 1.0)
end


@inline function vartype(T1::Type, T2::Type)
    @assert T1 <: T2 || T1 >: T2
    T = T1 <: T2 ? T2 : T1
    return T
end


@inline function assertdim(x::AbstractArray, d::Int)
    D = ndims(x)
    @assert D==d "expected input-dim is $d but got $D"
end

@inline function assertdim(x::Variable, d::Int)
    D = ndims(x)
    @assert D==d "expected input-dim is $d but got $D"
end
