export ysize_and_xrange_when_pad
export padconst

const Dims2 = Tuple{Int, Int}


@inline function paddings(pads::Pads{D}) where D
    return sum( sum.(pads) )
end

@inline function paddings(pads::Vector{Dims2})
    return sum( sum.(pads) )
end

@inline function paddings(pads::Dims{D}) where D
    return sum(pads)
end

@inline function paddings(pads::Vector{Int})
    return sum(pads)
end


"""
    ysize_and_xrange_when_pad(x::AbstractArray, pads::Pads{D}) where D -> ysize, xrange

Suppose padding `x` to `y` according to `pads`, the returned arg has the following meaning
+ `ysize`::NTuple is the size of y
+ `xrange`::CartesianIndices is the range that we should insert x into y
# Example One
    julia> x = reshape(collect(1:12), (2,3,2));
    julia> p = ((1,1), (1,2));
    julia> ysize_and_xrange_when_pad(x, p)
    ((4, 6, 2), CartesianIndices((2:3, 2:4, 1:2)))
         ↑                              ↑
         ysize                          xrange
# Example Two
    julia> x = reshape(collect(1:6), (2,3));
    julia> ysize_and_xrange_when_pad( x, ( (1,2), (1,3) ) )
    (5, 7), CartesianIndices((2:3, 2:4))
        ┌───┬───┬───┐
        │ 1 │ 3 │ 5 │
        ├───┼───┼───┤
        │ 2 │ 4 │ 6 │
        └───┴───┴───┘
             ↓↓↓
    ┌───┬───┬───┬───┬───┬───┬───┐
    │   │   │   │   │   │   │   │ 1
    ├───┼───┼───┼───┼───┼───┼───┤
    │   │ 1 │ 3 │ 5 │   │   │   │ 2
    ├───┼───┼───┼───┼───┼───┼───┤
    │   │ 2 │ 4 │ 6 │   │   │   │ 3     padding = ((1,2), (1,3))
    ├───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │ 4
    ├───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │ 5
    └───┴───┴───┴───┴───┴───┴───┘
      1   2   3   4   5   6   7
"""
function ysize_and_xrange_when_pad(x::AbstractArray, pads::Pads{D}) where D
    sizex = size(x)
    N = length(sizex)
    @assert D ≤ N "too much padding dims"
    newsize = ntuple(i -> i > D ? 0 : pads[i][1] + pads[i][2], N) .+ sizex
    offsets = ntuple(i -> i > D ? 0 : pads[i][1],  N)
    xrange  = ntuple(i -> offsets[i] .+ axes(x,i), N)
    return newsize, CartesianIndices(xrange)
end

"""
    ysize_and_xrange_when_pad(x::AbstractArray, pads::Vector{Dims2})

# Example
    julia> x = reshape(collect(1:12), (2,3,2));
    julia> p = [(1,1),(1,2)];
    julia> ysize_and_xrange_when_pad(x, p)
    ((4, 6, 2), CartesianIndices((2:3, 2:4, 1:2)))
"""
function ysize_and_xrange_when_pad(x::AbstractArray, pads::Vector{Dims2})
    sizex = size(x)
    D = length(pads)
    N = length(sizex)
    @assert D ≤ N "too much padding dims"
    newsize = ntuple(i -> i > D ? 0 : pads[i][1] + pads[i][2], N) .+ sizex
    offsets = ntuple(i -> i > D ? 0 : pads[i][1],  N)
    xrange  = ntuple(i -> offsets[i] .+ axes(x,i), N)
    return newsize, CartesianIndices(xrange)
end


"""
    ysize_and_xrange_when_pad(x::AbstractArray, pads::Vector{Pair{Int,Dims2}})

# Example
    julia> x = reshape( collect(1:12), (2,3,2) );
    julia> p = [2=>(1,2),1=>(1,1)];
    julia> ysize_and_xrange_when_pad(x, p)
    ((4, 6, 2), CartesianIndices((2:3, 2:4, 1:2)))
"""
function ysize_and_xrange_when_pad(x::AbstractArray, pads::Vector{Pair{Int,Dims2}})
    P = length(pads)
    D = ndims(x)
    @assert P ≤ D "too much padding dims"

    xsize = size(x)
    start = Vector{Int}(undef, D)
    final = Vector{Int}(undef, D)
    ysize = Vector{Int}(undef, D)

    for i in 1:D
        start[i] = 1
        final[i] = xsize[i]
        ysize[i] = xsize[i]
    end

    for i in 1:length(pads)
        d = pads[i][1]      # stored dimension information
        t = pads[i][2]      # stored padding information
        p = t[1] + t[2]     # total length-increas after padding
        start[d] += t[1]    # start idx after padding
        final[d] += t[1]    # end   idx after padding
        ysize[d] += p       # new length after padding for d-th dimension
    end

    newsize = ntuple(i -> ysize[i],          D)
    xrange  = ntuple(i -> start[i]:final[i], D)
    return newsize, CartesianIndices(xrange)
end


"""
    padconst(x::AbstractArray, pads::Pads{D}, val::Real=0) where D

The argment `pads` is like ((1,2),(4,3)), which means:
+ for the `1-st` dim
    the padding is 1 at the begining and 2 at the end;
+ for the `2-nd` dim
    the padding is 4 at the begining and 3 at the end;
# Example
    julia> x = reshape(collect(1:12), 2,3,2);
    julia> padconst(x, ((1,1),(1,2)), 0)
    4×6×2 Array{Int64, 3}:
    [:, :, 1] =
     0  0  0  0  0  0
     0  1  3  5  0  0
     0  2  4  6  0  0
     0  0  0  0  0  0

    [:, :, 2] =
     0  0   0   0  0  0
     0  7   9  11  0  0
     0  8  10  12  0  0
     0  0   0   0  0  0
"""
function padconst(x::AbstractArray, pads::Pads{D}, val::Real=0) where D
    paddings(pads) == 0 && return x
    ysize, xrange = ysize_and_xrange_when_pad(x, pads)
    y = fill!(similar(x, ysize), val)
    y[xrange] = x
    return y
end

"""
    padconst(x::AbstractArray, pads::Vector{Dims2}, val::Real=0)

The argment `pads` is like [(1,2),(4,3)], which means:
+ for the `1-st` dim
    the padding is 1 at the begining and 2 at the end;
+ for the `2-nd` dim
    the padding is 4 at the begining and 3 at the end;
# Example
    julia> padconst(reshape( collect(1:12), (2,3,2) ), [(1,1),(1,2)], 0)
    4×6×2 Array{Int64, 3}:
    [:, :, 1] =
     0  0  0  0  0  0
     0  1  3  5  0  0
     0  2  4  6  0  0
     0  0  0  0  0  0

    [:, :, 2] =
     0  0   0   0  0  0
     0  7   9  11  0  0
     0  8  10  12  0  0
     0  0   0   0  0  0
"""
function padconst(x::AbstractArray, pads::Vector{Dims2}, val::Real=0)
    paddings(pads) == 0 && return x
    ysize, xrange = ysize_and_xrange_when_pad(x, pads)
    y = fill!(similar(x, ysize), val)
    y[xrange] = x
    return y
end


"""
    padconst(x::AbstractArray, pads::Vector{Pair{Int,Dims2}}, val::Real=0)

The argment `pads` is like [2=>(1,2),1=>(4,3)], which means:
+ for the `1-st` dim
    the padding is 4 at the begining and 3 at the end;
+ for the `2-nd` dim
    the padding is 1 at the begining and 2 at the end;
# Example
    julia> padconst(reshape( collect(1:12), (2,3,2) ), [2=>(1,2), 1=>(1,1)], 0)
    4×6×2 Array{Int64, 3}:
    [:, :, 1] =
     0  0  0  0  0  0
     0  1  3  5  0  0
     0  2  4  6  0  0
     0  0  0  0  0  0

    [:, :, 2] =
     0  0   0   0  0  0
     0  7   9  11  0  0
     0  8  10  12  0  0
     0  0   0   0  0  0
"""
function padconst(x::AbstractArray, pads::Vector{Pair{Int,Dims2}}, val::Real=0)
    paddings(pads) == 0 && return x
    ysize, xrange = ysize_and_xrange_when_pad(x, pads)
    y = fill!(similar(x, ysize), val)
    y[xrange] = x
    return y
end


function padconst(x::Variable{T}, pads::Pads{D}, val::Real=0) where {T,D}
    paddings(pads) == 0 && return x
    ysize, xrange = ysize_and_xrange_when_pad(x.value, pads)
    t = fill!(similar(x.value, ysize), val)
    t[xrange] = x.value

    y = Variable{T}(t, x.backprop)
    if y.backprop
        y.backward = function ∇padconst()
            if needgrad(x)
                x ← δ(y)[xrange]
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function padconst(x::Variable{T}, pads::Vector{Dims2}, val::Real=0) where T
    paddings(pads) == 0 && return x
    ysize, xrange = ysize_and_xrange_when_pad(x.value, pads)
    t = fill!(similar(x.value, ysize), val)
    t[xrange] = x.value

    y = Variable{T}(t, x.backprop)
    if y.backprop
        y.backward = function ∇padconst()
            if needgrad(x)
                x ← δ(y)[xrange]
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function padconst(x::Variable{T}, pads::Vector{Pair{Int,Dims2}}, val::Real=0) where T
    paddings(pads) == 0 && return x
    ysize, xrange = ysize_and_xrange_when_pad(x.value, pads)
    t = fill!(similar(x.value, ysize), val)
    t[xrange] = x.value

    y = Variable{T}(t, x.backprop)
    if y.backprop
        y.backward = function ∇padconst()
            if needgrad(x)
                x ← δ(y)[xrange]
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    range_when_unpad(xten::AbstractArray, pads::Pads{D}) where D
Suppose `x` was padded to `xten`, so we reverse inference the coords of `x` give `xten` and `pads` info
"""
function range_when_unpad(xten::AbstractArray, pads::Pads{D}) where D
    xsize = size(xten)
    N = length(xsize)
    @assert D ≤ N "too much padding dims"
    xrange = ntuple(N) do i
        i > D && return 1:xsize[i]
        return (pads[i][1]+1):(xsize[i] - pads[i][2])
    end
    return CartesianIndices(xrange)
end


function unpad(xten::AbstractArray, pads::Pads{D}) where D
    paddings(pads) == 0 && return xten
    range = range_when_unpad(xten, pads)
    return xten[range]
end


function unpad(xten::Variable, pads::Pads{D}) where D
    paddings(pads) == 0 && return xten
    range = range_when_unpad(ᵛ(xten), pads)
    return xten[range]
end
