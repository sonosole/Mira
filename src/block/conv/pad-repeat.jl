export padrepeat

"""
    padrepeat(x::AbstractArray, pads::Pads{D}) where D

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
function padrepeat(x::AbstractArray, pads::Pads{D}) where D
    paddings(pads) == 0 && return x
    ysize, xranges = ysize_and_xrange_when_pad(x, pads)
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


function padrepeat(x::Variable{T}, pads::Pads{D}) where {T,D}
    paddings(pads) == 0 && return x
    ysize, xranges = ysize_and_xrange_when_pad(ᵛ(x), pads)
    v = similar(ᵛ(x), ysize)
    y = Variable{T}(v, x.backprop)
    v[xranges] = ᵛ(x)

    N = ndims(y)
    for d = 1:D
        padleft  = pads[d][1]≠0
        padright = pads[d][2]≠0
        if padleft
            P = pads[d][1]                   # left paddings
            E = first(xranges.indices[d])    # left side edge
            dstᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (1:P), N)
            srcᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (E:E), N)
            ᵛ(y)[CartesianIndices(dstᵢ)] .= ᵛ(y)[CartesianIndices(srcᵢ)]
        end
        if padright
            P = pads[d][2]                   # right paddings
            E = last(xranges.indices[d])     # right side edge
            dstᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (1+E:E+P), N)
            srcᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (  E:E  ), N)
            ᵛ(y)[CartesianIndices(dstᵢ)] .= ᵛ(y)[CartesianIndices(srcᵢ)]
        end
    end

    if y.backprop
        y.backward = function ∇padrepeat()
            if needgrad(x)
                zerodelta(x)
                for d = 1:D
                    padleft  = pads[d][1]≠0
                    padright = pads[d][2]≠0
                    if padleft
                        P = pads[d][1]                   # left paddings
                        E = first(xranges.indices[d])    # left side edge
                        dstᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (1:P), N)
                        srcᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (E:E), N)
                        ᵟ(y)[CartesianIndices(srcᵢ)] += sum(ᵟ(y)[CartesianIndices(dstᵢ)], dims=d)
                    end
                    if padright
                        P = pads[d][2]                   # right paddings
                        E = last(xranges.indices[d])     # right side edge
                        dstᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (1+E:E+P), N)
                        srcᵢ = ntuple(k -> k≠d ? (1:ysize[k]) : (  E:E  ), N)
                        ᵟ(y)[CartesianIndices(srcᵢ)] += sum(ᵟ(y)[CartesianIndices(dstᵢ)], dims=d)
                    end
                end
                x ← ᵟ(y)[xranges]
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end
