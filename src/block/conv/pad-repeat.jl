export padrepeat

"""
    padrepeat(x::AbstractArray, pads::NTuple{D,Dims{2}}) where D

# Example
    julia> x = reshape(collect(1:6), 1,6)
    1×6 Matrix{Int64}:
     1  2  3  4  5  6

    julia> padrepeat(x, ((0,0),(2,3)) )
    1×11 Matrix{Int64}:
     [1  1]  1  2  3  4  5  6  [6  6  6]
             ↑              ↑
              copied points
"""
function padrepeat(x::AbstractArray, pads::NTuple{D,Dims2}) where D
    paddings(pads) == 0 && return x
    ysize, xranges = size_and_range(x, pads)
    y = similar(x, ysize)
    y[xranges] = x

    N = ndims(y)
    for d = 1:D
        padleft  = pads[d][1]≠0
        padright = pads[d][2]≠0
        if padleft
            P = pads[d][1]                   # left paddings
            E = first(xranges.indices[d])    # left side edge
            dstᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (1:P), N)
            srcᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (E:E), N)
            y[CartesianIndices(dstᵢ)] .= y[CartesianIndices(srcᵢ)]
        end
        if padright
            P = pads[d][2]                   # right paddings
            E = last(xranges.indices[d])     # right side edge
            dstᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (1+E:E+P), N)
            srcᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (  E:E  ), N)
            y[CartesianIndices(dstᵢ)] .= y[CartesianIndices(srcᵢ)]
        end
    end
    return y
end
