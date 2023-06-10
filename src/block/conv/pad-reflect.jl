export padreflect

"""
    padreflect(x::AbstractArray, pads::NTuple{D,Dims2}) where D

# Example
    julia> x = reshape(collect(1:6), 1,6)
    1×6 Matrix{Int64}:
     1  2  3  4  5  6

    julia> padreflect(x, ((0,0),(2,3)) )
    1×11 Matrix{Int64}:
     [3  2]  1  2  3  4  5  6  [5  4  3]
             ↑              ↑
              reflect points
"""
function padreflect(x::AbstractArray, pads::NTuple{D,Dims2}) where D
    paddings(pads) == 0 && return x
    ysize, xranges = size_and_range(x, pads)
    y = fill!(similar(x, ysize), 0)
    y[xranges] = x

    N = ndims(y)
    for d = 1:D
        padleft  = pads[d][1]≠0
        padright = pads[d][2]≠0
        if padleft
            P = min(pads[d][1], size(x,d)-1)  # left paddings
            E = first(xranges.indices[d])     # left side reflect edge
            dstᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (E-1:-1:E-P), N)
            srcᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (E+1  : E+P), N)
            y[CartesianIndices(dstᵢ)] .= y[CartesianIndices(srcᵢ)]
        end
        if padright
            P = min(pads[d][2], size(x,d)-1)  # right paddings
            E = last(xranges.indices[d])      # right side reflect edge
            dstᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (E+1  : E+P), N)
            srcᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (E-1:-1:E-P), N)
            y[CartesianIndices(dstᵢ)] .= y[CartesianIndices(srcᵢ)]
        end
    end
    return y
end
