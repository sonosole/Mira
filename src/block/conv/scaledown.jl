export drange
export scaledown
export scaledown1d
export scaledown2d
export scaledown3d


@inline function assertlength(d::Vector{Int}, i::Int)
    N = length(d)
    @assert N==i "expected scale-down-dim is $i but got $N"
end


function drange(x::AbstractArray, d::NTuple{D, Int}) where D
    N = ndims(x)
    @assert D ≤ N "too much scale-down dims"
    S = size(x)
    return ntuple(i -> (i > D) ? (1:1:S[i]) : (1:d[i]:S[i]), N)
end

function drange(x::AbstractArray, d::Vector{Int})
    N = ndims(x)
    D = length(d)
    @assert D ≤ N "too much scale-down dims"
    S = size(x)
    return ntuple(i -> (i > D) ? (1:1:S[i]) : (1:d[i]:S[i]), N)
end


"""
    scaledown(x::AbstractArray, d::NTuple)
# Example
    julia> x = reshape(collect(1:15),3,5)
    3×5 Matrix{Int64}:
     1  4  7  10  13
     2  5  8  11  14
     3  6  9  12  15

    julia> y = scaledown(x, (2,3))
    2×2 Matrix{Int64}:
     1  10
     3  12
"""
function scaledown(x::AbstractArray, d::NTuple{D, Int}) where D
    prod(d) == 1 && return x
    return x[drange(x, d)...]
end

function scaledown(x::AbstractArray, d::Vector{Int})
    prod(d) == 1 && return x
    return x[drange(x, d)...]
end

"""
    scaledown(x::Variable, d::NTuple)
# Example
    julia> x = Variable(reshape(collect(1:15),3,5), keepsgrad=true)
     Leaf's value is 3×5 Matrix{Float32}:
     1.0  4.0  7.0  10.0  13.0
     2.0  5.0  8.0  11.0  14.0
     3.0  6.0  9.0  12.0  15.0

    julia> y = scaledown(x, (2,3))
     Leaf's value is 2×2 Matrix{Float32}:
     1.0  10.0
     3.0  12.0

    julia> backward(y)
    julia> x
     Leaf's value is 3×5 Matrix{Float32}:
     1.0  4.0  7.0  10.0  13.0
     2.0  5.0  8.0  11.0  14.0
     3.0  6.0  9.0  12.0  15.0
     Leaf's delta is 3×5 Matrix{Float32}:
     1.0  0.0  0.0  1.0  0.0
     0.0  0.0  0.0  0.0  0.0
     1.0  0.0  0.0  1.0  0.0
"""
function scaledown(x::Variable, d::NTuple{D, Int}) where D
    prod(d) == 1 && return x
    return x[drange(ᵛ(x), d)...]
end

function scaledown(x::Variable, d::Vector{Int})
    prod(d) == 1 && return x
    return x[drange(ᵛ(x), d)...]
end

"""
    scaledown1d(x::AbstractArray, d::Int)
Applied to batched multi-channel `1-D` data of shape (Channels, `Length`, Batchsize)
"""
function scaledown1d(x::AbstractArray, d::Int)
    assertdim(x, 3)
    d==1 && return x
    dilation = (1, d, 1)
    return x[drange(x, dilation)...]
end

"""
    scaledown2d(x::AbstractArray, d::Int)
Applied to batched multi-channel `2-D` data of shape (Channels, `Hight`, `Width`, Batchsize)
"""
function scaledown2d(x::AbstractArray, d::Int)
    assertdim(x, 4)
    d==1 && return x
    dilation = (1, d, d, 1)
    return x[drange(x, dilation)...]
end

"""
    scaledown2d(x::AbstractArray, d1::Int, d2::Int)
Applied to batched multi-channel `2-D` data of shape (Channels, `Hight`, `Width`, Batchsize)
"""
function scaledown2d(x::AbstractArray, d1::Int, d2::Int)
    assertdim(x, 4)
    d1*d2==1 && return x
    dilation = (1, d1, d2, 1)
    return x[drange(x, dilation)...]
end

"""
    scaledown2d(x::AbstractArray, d::NTuple{2,Int})
Applied to batched multi-channel `2-D` data of shape (Channels, `Hight`, `Width`, Batchsize)
"""
function scaledown2d(x::AbstractArray, d::NTuple{2,Int})
    assertdim(x, 4)
    prod(d)==1 && return x
    dilation = (1, d[1], d[2], 1)
    return x[drange(x, dilation)...]
end

"""
    scaledown2d(x::AbstractArray, d::Vector{Int})
Applied to batched multi-channel `2-D` data of shape (Channels, `Hight`, `Width`, Batchsize)
"""
function scaledown2d(x::AbstractArray, d::Vector{Int})
    assertdim(x, 4)
    assertlength(d, 2)
    prod(d)==1 && return x
    dilation = (1, d[1], d[2], 1)
    return x[drange(x, dilation)...]
end


"""
    scaledown3d(x::AbstractArray, d::Int)
Applied to batched multi-channel `3-D` data of shape (Channels, `Hight`, `Width`, `Depth`, Batchsize)
"""
function scaledown3d(x::AbstractArray, d::Int)
    assertdim(x, 5)
    d==1 && return x
    dilation = (1, d, d, d, 1)
    return x[drange(x, dilation)...]
end

"""
    scaledown3d(x::AbstractArray, d1::Int, d2::Int, d3::Int)
Applied to batched multi-channel `3-D` data of shape (Channels, `Hight`, `Width`, `Depth`, Batchsize)
"""
function scaledown3d(x::AbstractArray, d1::Int, d2::Int, d3::Int)
    assertdim(x, 5)
    d1*d2*d3==1 && return x
    dilation = (1, d1, d2, d3, 1)
    return x[drange(x, dilation)...]
end

"""
    scaledown3d(x::AbstractArray, d::NTuple{3,Int})
Applied to batched multi-channel `3-D` data of shape (Channels, `Hight`, `Width`, `Depth`, Batchsize)
"""
function scaledown3d(x::AbstractArray, d::NTuple{3,Int})
    assertdim(x, 5)
    prod(d)==1 && return x
    dilation = (1, d[1], d[2], d[3], 1)
    return x[drange(x, dilation)...]
end

"""
    scaledown3d(x::AbstractArray, d::Vector{Int})
Applied to batched multi-channel `3-D` data of shape (Channels, `Hight`, `Width`, `Depth`, Batchsize)
"""
function scaledown3d(x::AbstractArray, d::Vector{Int})
    assertdim(x, 5)
    assertlength(d, 3)
    prod(d)==1 && return x
    dilation = (1, d[1], d[2], d[3], 1)
    return x[drange(x, dilation)...]
end


"""
    scaledown1d(x::Variable, d::Int)
Applied to batched multi-channel `1-D` Variable of shape (Channels, `Length`, Batchsize)
"""
function scaledown1d(x::Variable, d::Int)
    assertdim(x, 3)
    d==1 && return x
    dilation = (1, d, 1)
    return x[drange(ᵛ(x), dilation)...]
end


"""
    scaledown2d(x::Variable, d::Int)
Applied to batched multi-channel `2-D` Variable of shape (Channels, `Hight`, `Width`, Batchsize)
"""
function scaledown2d(x::Variable, d::Int)
    assertdim(x, 4)
    d==1 && return x
    dilation = (1, d, d, 1)
    return x[drange(ᵛ(x), dilation)...]
end

"""
    scaledown2d(x::Variable, d1::Int, d2::Int)
Applied to batched multi-channel `2-D` Variable of shape (Channels, `Hight`, `Width`, Batchsize)
"""
function scaledown2d(x::Variable, d1::Int, d2::Int)
    assertdim(x, 4)
    d1*d2==1 && return x
    dilation = (1, d1, d2, 1)
    return x[drange(ᵛ(x), dilation)...]
end

"""
    scaledown2d(x::Variable, d::NTuple{2,Int})
Applied to batched multi-channel `2-D` Variable of shape (Channels, `Hight`, `Width`, Batchsize)
"""
function scaledown2d(x::Variable, d::NTuple{2,Int})
    assertdim(x, 4)
    prod(d)==1 && return x
    dilation = (1, d[1], d[2], 1)
    return x[drange(ᵛ(x), dilation)...]
end

"""
    scaledown2d(x::Variable, d::Vector{Int})
Applied to batched multi-channel `2-D` Variable of shape (Channels, `Hight`, `Width`, Batchsize)
"""
function scaledown2d(x::Variable, d::Vector{Int})
    assertdim(x, 4)
    assertlength(d, 2)
    prod(d)==1 && return x
    dilation = (1, d[1], d[2], 1)
    return x[drange(ᵛ(x), dilation)...]
end


"""
    scaledown3d(x::Variable, d::Int)
Applied to batched multi-channel `3-D` Variable of shape (Channels, `Hight`, `Width`, `Depth`, Batchsize)
"""
function scaledown3d(x::Variable, d::Int)
    assertdim(x, 5)
    d==1 && return x
    dilation = (1, d, d, d, 1)
    return x[drange(ᵛ(x), dilation)...]
end

"""
    scaledown3d(x::Variable, d1::Int, d2::Int, d3::Int)
Applied to batched multi-channel `3-D` Variable of shape (Channels, `Hight`, `Width`, `Depth`, Batchsize)
"""
function scaledown3d(x::Variable, d1::Int, d2::Int, d3::Int)
    assertdim(x, 5)
    d1*d2*d3==1 && return x
    dilation = (1, d1, d2, d3, 1)
    return x[drange(ᵛ(x), dilation)...]
end

"""
    scaledown3d(x::Variable, d::NTuple{3,Int})
Applied to batched multi-channel `3-D` Variable of shape (Channels, `Hight`, `Width`, `Depth`, Batchsize)
"""
function scaledown3d(x::Variable, d::NTuple{3,Int})
    assertdim(x, 5)
    prod(d)==1 && return x
    dilation = (1, d[1], d[2], d[3], 1)
    return x[drange(ᵛ(x), dilation)...]
end

"""
    scaledown3d(x::Variable, d::Vector{Int})
Applied to batched multi-channel `3-D` Variable of shape (Channels, `Hight`, `Width`, `Depth`, Batchsize)
"""
function scaledown3d(x::Variable, d::Vector{Int})
    assertdim(x, 5)
    assertlength(d, 3)
    prod(d)==1 && return x
    dilation = (1, d[1], d[2], d[3], 1)
    return x[drange(ᵛ(x), dilation)...]
end
