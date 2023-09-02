export divchannel
export catchannel


"""
# Example
    julia> x = reshape(collect(1:6*2),6,2)
    8×2 Matrix{Int64}:
     1   9
     2  10
     3  11
     4  12
     5  13
     6  14

    julia> xs = divchannel(x, 2);

    julia> xs[1]
    3×2 Matrix{Int64}:
     1  7
     2  8
     3  9
    julia> xs[2]
    3×2 Matrix{Int64}:
     4  10
     5  11
     6  12
"""
function divchannel(x::AbstractArray, G::Int)
    xsize = size(x)
    C = first(xsize)  # all groups' channels
    E = div(C, G)     # one group's channels

    I = ntuple(i -> 1:xsize[i+1], ndims(x)-1)
    postidx = CartesianIndices(I)

    X = Vector{AbstractArray}(undef, G)
    Threads.@threads for i in 1:G
        offset = (i-1) * E
        preidx = 1 + offset : E + offset
        X[i] = x[preidx, postidx]
    end

    return X
end


"""
# Example
    julia> x  = Variable(reshape(collect(1:8*2),8,2), keepsgrad=true);
    julia> xs = divchannel(x, 2)
    2-element Vector{Variable}:
      None Leaf's value is 4×2 Matrix{Float32}:
     1.0   9.0
     2.0  10.0
     3.0  11.0
     4.0  12.0

      None Leaf's value is 4×2 Matrix{Float32}:
     5.0  13.0
     6.0  14.0
     7.0  15.0
     8.0  16.0

    julia> backward(xs[1], 2);
    julia> backward(xs[2], 9);

    julia> x.delta
    8×2 Matrix{Float32}:
     2.0  2.0
     2.0  2.0
     2.0  2.0
     2.0  2.0
     9.0  9.0
     9.0  9.0
     9.0  9.0
     9.0  9.0
"""
function divchannel(x::Variable{T}, G::Int) where T
    xsize = size(x)
    C = first(xsize)  # all groups' channels
    E = div(C, G)     # one group's channels

    I = ntuple(i -> 1:xsize[i+1], ndims(x)-1)
    postidx = CartesianIndices(I)

    X = Vector{Variable{T}}(undef, G)
    Threads.@threads for i in 1:G
        offset = (i-1) * E
        preidx = 1 + offset : E + offset
        X[i] = x[preidx, postidx]
    end

    return X
end


function catchannel(xs::Vector{AbstractArray})
    x = first(xs)
    xsize = size(x)
    C = first(xsize)
    G = length(xs)
    y = similar(x, ntuple(i -> i>1 ? xsize[i] : G*C, ndims(x)))
    postidx = CartesianIndices(ntuple(i -> 1:xsize[i+1], ndims(x)-1))

    Threads.@threads for i in 1:G
        offset = (i-1) * C
        preidx = 1 + offset : C + offset
        y[preidx, postidx] .= xs[i]
    end
    return y
end


"""
# Example
    julia> x  = Variable(reshape(collect(1:8*2),8,2), keepsgrad=true);
    julia> xs = divchannel(x, 2);
    julia> y  = catchannel(xs);
"""
function catchannel(xs::Vector{Variable{T}}) where T
    x = first(xs)
    xsize = size(x)
    C = first(xsize)    # nchannels
    G = length(xs)      # groups
    v = similar(ᵛ(x), ntuple(i -> i>1 ? xsize[i] : G*C, ndims(x)))
    postidx = CartesianIndices(ntuple(i -> 1:xsize[i+1], ndims(x)-1))

    Threads.@threads for i in 1:G
        offset = (i-1) * C
        preidx = 1 + offset : C + offset
        v[preidx, postidx] .= value(xs[i])
    end

    y = Variable{T}(v, x.backprop)

    if y.backprop
        y.backward = function ∇catchannel()
            if need2computeδ!(x)
                Threads.@threads for i in 1:G
                    offset = (i-1) * C
                    preidx = 1 + offset : C + offset
                    xs[i] ← y.delta[preidx, postidx]
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        for i in 1:G
            addchild(y, xs[i])
        end
    end

    return y
end
