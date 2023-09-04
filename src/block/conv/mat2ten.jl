export mat2ten
export infer_conv_out_size
export infer_tconv_out_size

"""
    infer_conv_out_size(x::Array, padding,kernel,dilation,stride,zchannels) -> zsize::Dims

Infer Conv operation's output size, if `z` = Conv(`x`), then `zsize` = size(`z`). zchannels = `zsize`[1]
This function is just for checking, not for training or inferencing.

# Example
```julia
padding   = ((1,3), (1,3))
kernel    = (2,3)
dilation  = (3,2)
stride    = (1,2)
zchannels = 1
x     = reshape([i for i in 1:6], 1,2,3,1);
zsize = infer_conv_out_size(x, padding, kernel, dilation, stride, zchannels)
```
    ┌───┬───┬───┐
    │ 1 │ 3 │ 5 │
    ├───┼───┼───┤
    │ 2 │ 4 │ 6 │
    └───┴───┴───┘ 2×3
         ↓↓↓  padding = ((1,3), (1,3))
    ┌───┬───┬───┬───┬───┬───┬───┐
    │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┤
    │   │ 1 │ 3 │ 5 │   │   │   │                         ┌───┬───┐
    ├───┼───┼───┼───┼───┼───┼───┤       kernel = (2,3)    │ 1 │ 4 │
    │   │ 2 │ 4 │ 6 │   │   │   │     dilation = (3,2)    ├───┼───┤
    ├───┼───┼───┼───┼───┼───┼───┤       stride = (1,2)    │ 2 │ 5 │
    │   │   │   │   │   │   │   │     ────────────────►   ├───┼───┤
    ├───┼───┼───┼───┼───┼───┼───┤                         │ 3 │ 6 │
    │   │   │   │   │   │   │   │                         └───┴───┘ 3×2
    ├───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │
    └───┴───┴───┴───┴───┴───┴───┘ 6×7
"""
function infer_conv_out_size(x         :: AbstractArray,
                             padding   :: Pads{D},
                             kernel    :: Dims{D},
                             dilation  :: Dims{D},
                             stride    :: Dims{D},
                             zchannels :: Int) where D
    # In short, the normal convolution is Z = Conv(X), decomposed into following:
    #    ┌─────────────────────────────────────────────────────────────────┐
    # X →│ [padfn] → Xten → [ten2mat] → Xmat → [W*(∙) + B] → Y → [reshape] │→ Z
    #    └─────────────────────────────────────────────────────────────────┘
    # It's a one-to-one mapping from size(X) to size(Z).
    N = D + 2
    assertdim(x, N)
    xsize   = size(x)
    width   = ntuple(i -> xsize[i+1] + sum(padding[i]), D)       # spatial width after padding
    ekernel = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D)  # equivalent kernel widths
    zsize   = ntuple(N) do j
        isequal(j, 1) && return zchannels  # channels of z
        isequal(j, N) && return xsize[j]   # batchsize
        i = j - 1
        return (width[i] - ekernel[i]) ÷ stride[i] + 1
    end
    return zsize
end


"""
    infer_tconv_out_size(z, padding,kernel,dilation,stride) -> xtensize::Dims

Infer Transpose Conv operation's output size, if `x` = TransConv(`z`), then xsize = size(`x`).

# Detailed Explanation
Normal Conv's Output size is
`O = {I - [D*(K-1) + 1]}/S + 1`
of which `D` is dialetion, `K` is equivalent kernel width, `S` is stride. So the input spatial width is
`I = (O - 1) * S + D*(K-1) + 1`
# Example
```julia
padding   = ((1,3), (1,3))
kernel    = (2,3)
dilation  = (3,2)
stride    = (1,2)
xchannels = 15
zchannels = 9
batchsize = 12
# conv size calc
x = reshape([i for i in 1:xchannels*2*3batchsize], xchannels,2,3,batchsize);
zsize = infer_conv_out_size(x, padding, kernel, dilation, stride, zchannels)

# trans-conv size calc
z = reshape([i for i in 1:prod(zsize)], (zsize));
xsize = infer_tconv_out_size(z, kernel, dilation, stride, xchannels)
```
    ┌───┬───┬───┐
    │ 1 │ 3 │ 5 │
    ├───┼───┼───┤
    │ 2 │ 4 │ 6 │
    └───┴───┴───┘ 2×3
         ↑↑↑  padding = ((1,3), (1,3))
    ┌───┬───┬───┬───┬───┬───┬───┐
    │   │   │   │   │   │   │   │
    ├───┼───┼───┼───┼───┼───┼───┤
    │   │ 1 │ 3 │ 5 │   │   │   │                         ┌───┬───┐
    ├───┼───┼───┼───┼───┼───┼───┤       kernel = (2,3)    │ 1 │ 4 │
    │   │ 2 │ 4 │ 6 │   │   │   │     dilation = (3,2)    ├───┼───┤
    ├───┼───┼───┼───┼───┼───┼───┤       stride = (1,2)    │ 2 │ 5 │
    │   │   │   │   │   │   │   │     ◄────────────────   ├───┼───┤
    ├───┼───┼───┼───┼───┼───┼───┤                         │ 3 │ 6 │
    │   │   │   │   │   │   │   │                         └───┴───┘ 3×2
    ├───┼───┼───┼───┼───┼───┼───┤
    │   │   │   │   │   │   │   │
    └───┴───┴───┴───┴───┴───┴───┘ 6×7
"""
function infer_tconv_out_size(z         :: AbstractArray,
                              kernel    :: Dims{D},
                              dilation  :: Dims{D},
                              stride    :: Dims{D},
                              xchannels :: Int) where D
    # The transpose convolution is X = TransConv(Z), decomposed into following:
    #    ┌──────────────────────────────────────────────────────────────────┐
    #    │ ┌────────────────────────┐         ┌───────────┐       ┌───────┐ │
    # X ←│ │[unpad] ← Xten ← [toten]│← Xmat ← │ W*(∙) + B │ ← Y ← │reshape│ │← Z
    #    │ └─────── mat2ten ────────┘         └───────────┘       └───────┘ │
    #    └─────────────────────────────TransConv────────────────────────────┘
    # It's a one-to-many mapping from size(Z) to size(X) when stride ≠ 1, so the
    # size return by this function maybe not precise, but a least guarantee.
    N = D + 2
    assertdim(z, N)
    zsize  = size(z)
    zwidth = ntuple(i -> zsize[i+1], D)    # spatial width of z
    return ntuple(N) do j                  # size of xten
        isequal(j, 1) && return xchannels  # channels of x
        isequal(j, N) && return zsize[j]   # batchsize
        i = j - 1
        return (zwidth[i] - 1) * stride[i] + dilation[i] * (kernel[i] - 1) + 1
    end
end


function infer_tconv_out_size(zsize     :: Dims{N},
                              kernel    :: Dims{D},
                              dilation  :: Dims{D},
                              stride    :: Dims{D},
                              xchannels :: Int) where {N,D}
    # The transpose convolution is X = TransConv(Z), decomposed into following:
    #    ┌──────────────────────────────────────────────────────────────────┐
    #    │ ┌────────────────────────┐         ┌───────────┐       ┌───────┐ │
    # X ←│ │[unpad] ← Xten ← [toten]│← Xmat ← │ W*(∙) + B │ ← Y ← │reshape│ │← Z
    #    │ └────────mat2ten─────────┘         └───Dense───┘       └───────┘ │
    #    └────────────────────────────TransConv─────────────────────────────┘
    # It's a one-to-many mapping from size(Z) to size(X) when stride ≠ 1, so the
    # size return by this function maybe not precise, but a least guarantee.
    @assert isequal(N, D+2) "dims mismatch, $N≠$(D+2)"
    zwidth = ntuple(i -> zsize[i+1], D)    # spatial width of z
    return ntuple(N) do j                  # size of xten, i.e. size after padding
        isequal(j, 1) && return xchannels  # channels of x
        isequal(j, N) && return zsize[j]   # batchsize
        i = j - 1
        return (zwidth[i] - 1) * stride[i] + dilation[i]*(kernel[i] - 1) + 1
    end
end

@inline function size_after_padded(xsize::Dims{N}, padding::Pads{N}) where N
    return ntuple(i -> xsize[i] + sum(padding[i]), N)
end

@inline function size_before_padded(paddedxsize::Dims{N}, padding::Pads{N}) where N
    return ntuple(i -> paddedxsize[i] - sum(padding[i]), N)
end

function ten2matFwdInfo(sizeofx  :: Dims{N},
                        padding  :: Pads{D},
                        kernel   :: Dims{D},
                        dilation :: Dims{D},
                        stride   :: Dims{D}) where {N,D}

    @assert isequal(N, D+2) "dims mismatch, $N≠$(D+2)"
    xwidth  = ntuple(i -> sizeofx[i+1] + sum(padding[i]), D)           # equivalent spatial width
    ekernel = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D)        # equivalent kernel sizes
    zwidth  = ntuple(i -> (xwidth[i] - ekernel[i]) ÷ stride[i] + 1, D) # equivalent moving steps, i.e. output feature spatial width

    xchannels = sizeofx[1]
    batchsize = sizeofx[N]
    npatches  = prod(zwidth)           # total moving steps along all D dims
    ROWS = xchannels * prod(kernel)    # total number of elements of a patch
    COLS = batchsize * npatches        # total moving steps in a batch
    Iter = Ten2matFwdIter(ekernel, dilation, stride, zwidth, ROWS, COLS, npatches, xchannels, batchsize)

    return Iter
end


"""
    mat2ten
A part of TransConv module
# Explain
The transpose convolution is X = TransConv(Z), decomposed into following:
```julia
   ┌──────────────────────────────────────────────────────────────────┐
   │ ┌────────────────────────┐         ┌───────────┐       ┌───────┐ │
X ←│ │[unpad] ← Xten ← [toten]│← Xmat ← │ W*(∙) + B │ ← Y ← │reshape│ │← Z
   │ └────────mat2ten─────────┘         └───Dense───┘       └───────┘ │
   └──────────────────────────────────────────────────────────────────┘
```
"""
function mat2ten(xmat     :: Array{T},
                 zsize    :: Dims{N},
                 xsize    :: DimsOrNil,
                 padding  :: Pads{D},
                 kernel   :: Dims{D},
                 dilation :: Dims{D},
                 stride   :: Dims{D}) where {T,N,D}

    xchannels = size(xmat, 1) ÷ prod(kernel)
    ndpadding = extendpad(padding)
    if isnothing(xsize)
        paddedxsize = infer_tconv_out_size(zsize, kernel, dilation, stride, xchannels)
        xsize = size_before_padded(paddedxsize, ndpadding)
    else
        paddedxsize = size_after_padded(xsize, ndpadding)
    end
    xten = Zeros(Array{T}, paddedxsize)

    FwdIter = ten2matFwdInfo(xsize, padding, kernel, dilation, stride)
    parallizable = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D) .≤ stride

    if !all(parallizable)
        BwdIter = Ten2matBwdIter(FwdIter.zwidth, parallizable)
        for pindices in BwdIter
            Threads.@threads for coords in pindices
                # locally parallel calculation
                n = coords2nth(BwdIter.zwidth, coords)
                m, t = FwdIter[n]
                @inbounds xten[t] .+= reshape(xmat[m], size(xten[t]))
            end
        end
    else
        Threads.@threads for (m, t) in FwdIter
            # globally parallel calculation
            @inbounds xten[t] .+= reshape(xmat[m], size(xten[t]))
        end
    end
    return unpad(xten, ndpadding)
end

function mat2ten(xmat     :: Variable{Array{T}},
                 zsize    :: Dims{N},
                 xsize    :: DimsOrNil,
                 padding  :: Pads{D},
                 kernel   :: Dims{D},
                 dilation :: Dims{D},
                 stride   :: Dims{D}) where {T,N,D}

    xchannels = size(xmat, 1) ÷ prod(kernel)
    ndpadding = extendpad(padding)
    if isnothing(xsize)
        paddedxsize = infer_tconv_out_size(zsize, kernel, dilation, stride, xchannels)
        xsize = size_before_padded(paddedxsize, ndpadding)
    else
        paddedxsize = size_after_padded(xsize, ndpadding)
    end
    xten = Zeros(Array{T}, paddedxsize)

    FwdIter = ten2matFwdInfo(xsize, padding, kernel, dilation, stride)
    parallizable = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D) .≤ stride

    if !all(parallizable)
        BwdIter = Ten2matBwdIter(FwdIter.zwidth, parallizable)
        for pindices in BwdIter
            Threads.@threads for coords in pindices
                # locally parallel calculation
                n = coords2nth(BwdIter.zwidth, coords)
                m, t = FwdIter[n]
                @inbounds xten[t] .+= reshape(xmat.value[m], size(xten[t]))
            end
        end
    else
        Threads.@threads for (m, t) in FwdIter
            # globally parallel calculation
            @inbounds xten[t] .+= reshape(xmat.value[m], size(xten[t]))
        end
    end

    xten = Variable{Array{T}}(xten, xmat.backprop)

    if xten.backprop
        batchsize = zsize[N]
        rows = size(xmat, 1)
        xten.backward = function ∇mat2ten()
            if needgrad(xmat)
                zerodelta(xmat)
                Threads.@threads for (m, t) in FwdIter
                    @inbounds xmat.delta[m] .= reshape(xten.delta[t], rows, batchsize)
                end
            end
            ifNotKeepδThenFreeδ!(xten)
        end
        addchild(xten, xmat)
    end

    return unpad(xten, ndpadding)
end
