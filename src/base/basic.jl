export Zeros
export Ones

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
