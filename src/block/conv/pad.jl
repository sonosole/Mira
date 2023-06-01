export size_and_range
export padconst


"""
    size_and_range(x::AbstractArray, padinfo::Vector{Tuple{Int, Int}})

# Example
    julia> size_and_range(reshape( collect(1:12), (2,3,2) ), [(1,1),(1,2)])
    ( (4, 6, 2), (2:3, 2:4, 1:2) )
"""
function size_and_range(x::AbstractArray, padinfo::Vector{Tuple{Int, Int}})
    sizex = size(x)
    D = length(padinfo)
    N = length(size(x))
    @assert D ≤ N "too much padding dims"
    newsize = ntuple(i -> i > D ? 0 : padinfo[i][1] + padinfo[i][2], N) .+ sizex
    xranges = broadcast((a,b) -> a .+ b, axes(x), ntuple(i -> i > D ? 0 : padinfo[i][1], N))
    return (newsize, xranges)
end


"""
    size_and_range(x::AbstractArray, padinfo::Vector{Pair{Int,Tuple{Int, Int}}})

# Example
    julia> size_and_range(reshape( collect(1:12), (2,3,2) ), [2=>(1,2),1=>(1,1)])
    ( (4, 6, 2), (2:3, 2:4, 1:2) )
"""
function size_and_range(x::AbstractArray, padinfo::Vector{Pair{Int,Tuple{Int, Int}}})
    D     = ndims(x)
    xsize = size(x)

    start = Vector{Int}(undef, D)
    final = Vector{Int}(undef, D)
    ysize = Vector{Int}(undef, D)

    for i in 1:D
        start[i] = 1
        final[i] = xsize[i]
        ysize[i] = xsize[i]
    end

    for i in 1:length(padinfo)
        d = padinfo[i][1]   # stored dimension information
        t = padinfo[i][2]   # stored padding information
        p = t[1] + t[2]     # total length-increas after padding
        start[d] += t[1]    # start idx after padding
        final[d] += t[1]    # end   idx after padding
        ysize[d] += p       # new length after padding for d-th dimension
    end

    return ntuple(i -> ysize[i], D), ntuple(i -> start[i]:final[i], D)
end


"""
    padconst(x::AbstractArray, padinfo::Vector{Tuple{Int, Int}}, val::Real=0)

The argment `padinfo` is like [(1,2),(4,3)], which means:
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
function padconst(x::AbstractArray, padinfo::Vector{Tuple{Int, Int}}, val::Real=0)
    newsize, xranges = size_and_range(x, padinfo)
    y = fill!(similar(x, newsize), val)
    y[xranges...] = x
    return y
end


"""
    padconst(x::AbstractArray, padinfo::Vector{Pair{Int,Tuple{Int, Int}}}, val::Real=0)

The argment `padinfo` is like [2=>(1,2),1=>(4,3)], which means:
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
function padconst(x::AbstractArray, padinfo::Vector{Pair{Int,Tuple{Int, Int}}}, val::Real=0)
    newsize, xranges = size_and_range(x, padinfo)
    y = fill!(similar(x, newsize), val)
    y[xranges...] = x
    return y
end



function padconst(x::Variable{T}, padinfo::Vector{Tuple{Int, Int}}, val::Real=0) where T
    newsize, xranges = size_and_range(x.value, padinfo)
    t = fill!(similar(x.value, newsize), val)
    t[xranges...] = x.value

    y = Variable{T}(t, x.backprop)
    if y.backprop
        y.backward = function ∇padconst()
            if need2computeδ!(x)
                x ← δ(y)[xranges...]
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function padconst(x::Variable{T}, padinfo::Vector{Pair{Int,Tuple{Int, Int}}}, val::Real=0) where T
    newsize, xranges = size_and_range(x.value, padinfo)
    t = fill!(similar(x.value, newsize), val)
    t[xranges...] = x.value

    y = Variable{T}(t, x.backprop)
    if y.backprop
        y.backward = function ∇padconst()
            if need2computeδ!(x)
                x ← δ(y)[xranges...]
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end
