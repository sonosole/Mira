"""
    row_col_index(n::Int, rows::Int) -> (row, col)
Return the row and col index of the n-th element of a matrix having shape (`rows`, *)
"""
@inline function row_col_index(n::Int, rows::Int)
    n = n - 1
    r = mod(n, rows) + 1  # row index
    c = div(n, rows) + 1  # col index
    return (r, c)
end

"""
    row_col_batch_index(n::Int, rows::Int, cols::Int) -> (row, col, batch)
Return the row, col and batch index of the n-th element of a array having shape (`rows`, `col`, *)
"""
@inline function row_col_batch_index(n::Int, rows::Int, cols::Int)
    r, j = row_col_index(n, rows)
    c, b = row_col_index(j, cols)
    return (r, c, b)
end

"""
    patchcoords(n::Int, dims::NTuple{D,Int}) where D

Return the `n`-th element's coords in array `x` having shape of (W1,W2,W3,...,WD, Batchsize).
`dims` = (W1,W2,W3,...,WD).
"""
@inline function patchcoords(n::Int, dims::NTuple{D,Int}) where D
    coords = Vector{Int}(undef, D+1)
    i = 1   # iter index for `Idxs`
    j = n   # inital total elements
    d = D   # iter index for `dims` argument
    while d > 0
        coords[i], j = row_col_index(j, dims[i])
        i += 1
        d -= 1
    end
    coords[D+1] = j

    return coords
end

@inline function patchcoords(n::Int, dims::Vector{Int})
    D = length(dims)
    coords = Vector{Int}(undef, D+1)
    i = 1   # iter index for `Idxs`
    j = n   # inital total elements
    d = D   # iter index for `dims` argument
    while d > 0
        coords[i], j = row_col_index(j, dims[i])
        i += 1
        d -= 1
    end
    coords[D+1] = j

    return coords
end



struct ConvndFwdIterIndices{D}
    kernel    :: NTuple{D,Int}   # equivalent kernel sizes for D dims
    dilation  :: NTuple{D,Int}   # dilation of filter kernels
    stride    :: NTuple{D,Int}   # stride of filter kernels
    dims      :: NTuple{D,Int}   # shape of the output feature in n-dim conv
    rows      :: Int             # rows of the output in im2col algorithm
    cols      :: Int             # cols of the output in im2col algorithm
    channels  :: Int             # number of input channels
    function ConvndFwdIterIndices(kernel   :: NTuple{D,Int},
                                  dilation :: NTuple{D,Int},
                                  stride   :: NTuple{D,Int},
                                  dims     :: NTuple{D,Int},
                                  rows     :: Int,
                                  cols     :: Int,
                                  channels :: Int) where D
        new{D}(kernel, dilation, stride, dims, rows, cols, channels)
    end
end

function Base.show(io::IO, ::MIME"text/plain", c::ConvndFwdIterIndices{D}) where D
    println("ConvndFwdIterIndices{$D}")
    println("  ", c.kernel,   "\t → equivalent kernel size")
    println("  ", c.dilation, "\t → dilation")
    println("  ", c.stride,   "\t → stride")
    println("  ", c.dims,     "\t → size of output features")
    println("  ", c.rows,     "\t → rows in im2col")
    println("  ", c.cols,     "\t → cols in im2col")
    println("  ", c.channels, "\t → input channels")
end

function Base.show(io::IO, ::MIME"text/plain", c::ConvndFwdIterIndices{1})
    println("ConvndFwdIterIndices{1}")
    println("  ", c.kernel[1],   "\t → equivalent kernel size")
    println("  ", c.dilation[1], "\t → dilation")
    println("  ", c.stride[1],   "\t → stride")
    println("  ", c.dims[1],     "\t → size of output features")
    println("  ", c.rows,        "\t → rows in im2col")
    println("  ", c.cols,        "\t → cols in im2col")
    println("  ", c.channels,    "\t → input channels")
end


function Base.getindex(C::ConvndFwdIterIndices{D}, n::Int) where D
    coords = patchcoords(n, C.dims)
    xiters = ntuple(D+2) do i
        if isequal(i, 1)
            return 1 : C.channels
        end
        if i ≤ D+1
            j = i - 1
            istart = 1 : C.dilation[j] : C.kernel[j]
            offset = (coords[j] - 1) * C.stride[j]
            return istart .+ offset
        end
        return coords[D+1] : coords[D+1]
    end
    yiters = (1 : C.rows, n : n)
    return CartesianIndices(yiters), CartesianIndices(xiters)
end


Base.length(C::ConvndFwdIterIndices)     = C.cols
Base.lastindex(C::ConvndFwdIterIndices)  = C.cols
Base.firstindex(C::ConvndFwdIterIndices) = 1

function Base.iterate(C::ConvndFwdIterIndices, i::Int=1)
    if i ≤ C.cols
        return C[i], i+1
    else
        return nothing
    end
end


function im2col_nd_infos(x        :: AbstractArray,
                         padding  :: NTuple{D,NTuple{2,Int}},
                         kernel   :: NTuple{D,Int},
                         dilation :: NTuple{D,Int},
                         stride   :: NTuple{D,Int}) where D

    assertdim(x, D + 2)
    xsize = size(x)
    CHANNELS  = xsize[1]
    BATCHSIZE = xsize[ndims(x)]

    W = ntuple(i -> xsize[i+1] + sum(padding[i]), D)        # equivalent spatial width
    K = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D)   # equivalent kernel sizes
    S = ntuple(i -> (W[i] - K[i]) ÷ stride[i] + 1, D)       # equivalent moving steps, i.e. output feature size

    STEPS  = prod(S)         # total moving steps along all D dims
    KERNEL = prod(kernel)    # total effective kernel size  along all D dims

    ROWS = CHANNELS * KERNEL    # total number of elements of a patch
    COLS = STEPS * BATCHSIZE    # total moving steps in a batch
    Iter = ConvndFwdIterIndices(K, dilation, stride, S, ROWS, COLS, CHANNELS)

    return ROWS, COLS, Iter
end


function im2col_1d_infos(x        :: AbstractArray,
                         padding  :: NTuple{2,Int},
                         kernel   :: Int,
                         dilation :: Int,
                         stride   :: Int)

    assertdim(x, 3)
    xsize = size(x)
    CHANNELS  = xsize[1]
    BATCHSIZE = xsize[ndims(x)]

    W = ntuple(i -> xsize[2] + sum(padding), 1)       # equivalent spatial width
    K = ntuple(i -> dilation * (kernel - 1) + 1, 1)   # equivalent kernel sizes
    S = ntuple(i -> (W[1] - K[1]) ÷ stride + 1, 1)    # equivalent moving steps, i.e. output feature size

    STEPS  = prod(S)         # total moving steps along all 1 dim
    KERNEL = prod(kernel)    # total effective kernel size  along all 1 dim

    ROWS = CHANNELS * KERNEL    # total number of elements of a patch
    COLS = STEPS * BATCHSIZE    # total moving steps in a batch
    Iter = ConvndFwdIterIndices(K, (dilation,), (stride,), S, ROWS, COLS, CHANNELS)

    return ROWS, COLS, Iter
end


function im2col(x        :: Array{T},
                padding  :: NTuple{D,NTuple{2,Int}},
                kernel   :: NTuple{D,Int},
                dilation :: NTuple{D,Int},
                stride   :: NTuple{D,Int},
                padval   :: Real = 0) where {T,D}

    rows, cols, YXIndices = im2col_nd_infos(x, padding, kernel, dilation, stride)

    x = padconst(x, ntuple(i -> (1 < i < D+2) ? padding[i-1] : (0,0), D+2), padval)
    y = similar(x, rows, cols)

    Threads.@threads for (o, i) in YXIndices
        @inbounds y[o] .= reshape(x[i], rows, 1)
    end

    return y
end
