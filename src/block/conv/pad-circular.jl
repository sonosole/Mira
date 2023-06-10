export padcircular

"""
    padcircular(x::AbstractArray, pads::NTuple{D,Dims{2}}) where D

# Example
    julia> x = reshape(collect(1:6), 1,6)
    1×6 Matrix{Int64}:
     1  2  3  4  5  6

    julia> padcircular(x, ((0,0),(2,3)) )
    1×11 Matrix{Int64}:
     [5  6]  1  2  3  4  5  6  [1  2  3]
      ˈˈˈˈ   ¯¯¯¯¯¯¯     ˈˈˈˈ   ¯¯¯¯¯¯¯
            circular points
"""
function padcircular(x::AbstractArray, pads::NTuple{D,Dims2}) where D
    paddings(pads) == 0 && return x
    ysize, xranges = size_and_range(x, pads)
    y = fill!(similar(x, ysize), 0)
    y[xranges] = x

    N = ndims(y)
    for d = 1:D
        padleft  = pads[d][1]≠0
        padright = pads[d][2]≠0
        L = first(xranges.indices[d])    # left  side edge
        R =  last(xranges.indices[d])    # right side edge
        if padleft
            P = min(pads[d][1], size(x,d))
            dstᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (1:P),     N)
            srcᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (R-P+1:R), N)
            y[CartesianIndices(dstᵢ)] .= y[CartesianIndices(srcᵢ)]
        end
        if padright
            P = min(pads[d][2], size(x,d))
            dstᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (R+1:R+P), N)
            srcᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (L:L+P-1), N)
            y[CartesianIndices(dstᵢ)] .= y[CartesianIndices(srcᵢ)]
        end
    end
    return y
end
