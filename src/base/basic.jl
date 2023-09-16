export Zeros
export Ones
export vartype
export onehot


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


function onehot(i::Int, C::Int; type::Type=Array{Float32})
    x = Zeros(type, C, 1)
    x[i] = 1
    return x
end

"""
    onehot(labels::Vector{Int}, C::Int; type::Type=Array{Float32}) -> x::Type
+ `labels` is a batched labels, contains categories classification label
+ `C` is the nameber of categories, aka #dimentions of label
+ `type` is the returned var's type

# example
```
julia> onehot([1,2,3,4,3,2,1], 4, type=Array{Int})
4×7 Matrix{Int64}:
 1  0  0  0  0  0  1
 0  1  0  0  0  1  0
 0  0  1  0  1  0  0
 0  0  0  1  0  0  0
```
"""
function onehot(labels::Vector{Int}, C::Int; type::Type=Array{Float32})
    B = length(labels)
    x = Zeros(type, C, B)
    for (j, i) in enumerate(labels)
        x[i,j] = 1
    end
    return x
end
