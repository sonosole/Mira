"""
    infer_conv_out_size(x::Array, padding,kernel,dilation,stride) -> zsize::Dims
Infer Conv operation's output size, if `z` = Conv(`x`), then `zsize` = size(`z`). This function
is just for checking, not for training or inferencing.
# Example
```julia
padding = ((1,3), (1,3))
kernel  = (2,3)
dilation= (3,2)
stride  = (1,2)

# conv size calc
x = reshape([i for i in 1:6], 1,2,3,1);
ysize = infer_conv_out_size(x, padding, kernel, dilation, stride)
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
function infer_conv_out_size(x        :: AbstractArray,
                             padding  :: Mira.Pads{D},
                             kernel   :: Mira.Dims{D},
                             dilation :: Mira.Dims{D},
                             stride   :: Mira.Dims{D}) where D
    N = D + 2
    Mira.assertdim(x, N)
    shape   = size(x)
    xsize   = ntuple(i -> shape[i+1] + sum(padding[i]), D)           # spatial width after padding
    ekernel = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D)      # equivalent kernel widths
    zsize   = ntuple(N) do j
        isequal(j, 1) && return shape[1]
        isequal(j, N) && return shape[j]
        i = j - 1
        return (xsize[i] - ekernel[i]) ÷ stride[i] + 1
    end
    return zsize
end


"""
    infer_tconv_out_size(z, padding,kernel,dilation,stride) -> xsize::Dims
Infer Transpose Conv operation's output size, if `x` = TransConv(`z`), then xsize = size(`x`).
# Detailed Explanation
Normal Conv's Output size is
`O = {I - [D*(K-1) + 1]}/S + 1`
of which `D` is dialetion, `K` is equivalent kernel width, `S` is stride. So the input spatial width is
`I = (O - 1) * S + D*(K-1) + 1`
# Example
```julia
padding = ((1,3), (1,3))
kernel  = (2,3)
dilation= (3,2)
stride  = (1,2)

# conv size calc
x = reshape([i for i in 1:6], 1,2,3,1);
zsize = infer_conv_out_size(x, padding, kernel, dilation, stride)

# trans-conv size calc
z = reshape([i for i in 1:6], 1,3,2,1);
xsize = infer_tconv_out_size(z, kernel, dilation, stride)
```
"""
function infer_tconv_out_size(z        :: AbstractArray,
                              kernel   :: Mira.Dims{D},
                              dilation :: Mira.Dims{D},
                              stride   :: Mira.Dims{D}) where D
    N = D + 2
    Mira.assertdim(z, N)
    shape = size(z)
    zsize = ntuple(i -> shape[i+1], D)
    xsize = ntuple(N) do j
        isequal(j,1) && return shape[1]
        isequal(j,N) && return shape[j]
        i = j - 1
        return (zsize[i] - 1) * stride[i] + dilation[i]*(kernel[i] - 1) + 1
    end
    return xsize
end


function mat2ten(xmat     :: Variable{Array{T}},
                 xten     :: Array{T},
                 padding  :: Pads{D},
                 kernel   :: Dims{D},
                 dilation :: Dims{D},
                 stride   :: Dims{D},
                 padmode  :: Function = padconst,
                 padval   :: Real = 0) where {T,D}

    rows, cols, batchsize, YXIndices = ten2matFwdInfo(ᵛ(xten), padding, kernel, dilation, stride)
    parallizable = ntuple(i -> dilation[i] * (kernel[i] - 1) + 1, D) .≤ stride

    if !all(parallizable)
        BwdIter = Ten2matBwdIter(YXIndices.zsize, parallizable)
        for pindices in BwdIter
            # locally parallel calculation
            Threads.@threads for coords in pindices
                n = coords2nth(BwdIter.sizez, coords)
                o, i = YXIndices[n]
                @inbounds xten[i] .+= reshape(xmat.value[o], size(xten[i]))
            end
        end
    else
        # globally parallel calculation
        Threads.@threads for (o, i) in YXIndices
            @inbounds xten[i] .+= reshape(xmat.value[o], size(xten[i]))
        end
    end

    Xten = Variable{Array{T}}(xten, px.backprop)

    if Xten.backprop
        Xten.backward = function ∇mat2ten()
            if need2computeδ!(xmat)
                zerodelta(xmat)
                Threads.@threads for (o, i) in YXIndices
                    @inbounds xmat.delta[o] .= reshape(xten[i], rows, batchsize)
                end
            end
            ifNotKeepδThenFreeδ!(Xten)
        end
        addchild(Xten, xmat)
    end

    return Xten
end



"""
Applies a `D`-dim convolution over an `(D+2)`-dim input tensor of shape (ichannels, w1, w2, ..., wn, batchsize)\n
# Constructor
    TransConv{D}(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                 kernel   :: Dims{D} = ntuple(i -> 3, D),
                 dilation :: Dims{D} = ntuple(i -> 1, D),
                 stride   :: Dims{D} = ntuple(i -> 1, D),
                 padval   :: Real = 0f0,
                 padmode  :: String  = "zeros",
                 padding  :: PadsDOrStr = "valid",
                 type     :: Type = Array{Float32}) where D

+ `padmode` should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\", \"circular\"
+ `padding` can be \"valid\", \"same\", or type `NTuple{D, Dims{2}}`
"""
mutable struct TransConv{D} <: Block
    w :: VarOrNil
    b :: VarOrNil
    f :: FunOrNil
    kernel   :: Dims{D}
    dilation :: Dims{D}
    stride   :: Dims{D}
    padding  :: Pads{D}
    padmode  :: Function
    padval   :: Float32
    function TransConv{D}(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                          kernel   :: Dims{D} = ntuple(i -> 3, D),
                          dilation :: Dims{D} = ntuple(i -> 1, D),
                          stride   :: Dims{D} = ntuple(i -> 1, D),
                          padval   :: Real = 0f0,
                          padmode  :: String  = "zeros",
                          padding  :: PadsDOrStr = "valid",
                          type     :: Type = Array{Float32}) where D

        if padding isa String
            npads = inferpadding(padding, kernel, stride, dilation)
        elseif padding isa Int
            npads = ntuple(i -> (padding, padding), D)
        else
            npads = ntuple(D) do i
                if padding[i] isa Int
                    return (padding[i], padding[i])
                end
                if padding[i] isa Dims{2}
                    return padding[i]
                end
            end
        end

        dtype    = eltype(type)
        patchlen = prod(kernel) * ichannels
        Amplifer = dtype(sqrt(2 / patchlen))

        w = Amplifer * randn(dtype, ochannels, patchlen)
        b = Amplifer * randn(dtype, ochannels,        1)

        new{D}(Variable{type}(w,true,true,true),
               Variable{type}(b,true,true,true), fn,
               kernel, dilation, stride,
               npads, selectpad(padmode), padval)
    end
    function Conv{D}() where D
        O = (0, 0)
        new{D}(nothing, nothing, nothing,
               ntuple(i -> 3, D),
               ntuple(i -> 1, D),
               ntuple(i -> 1, D),
               ntuple(i -> O, D), padconst, 0f0)
    end
end


"""
+ Ordinary Conv Processes:
X → [padfn] → Xten → [ten2mat] → Xmat → [W*(●) + B] → Y → [reshape] → Z

+ Transpose Conv Processes:
X ← [unpad] ← Xten ← [mat2ten] ← Xmat ← [W*(●) + B] ← Y ← [reshape] ← Z
"""
function forward(C::TransConv{D}, Z::Variable{T}) where {T,D}
    W = C.w
    B = C.b
    N = D + 2
    Zchannels = size(Z, 1)
    batchsize = size(Z, N)
    xtensize = infer_tconv_out_size(ᵛ(Z), C.padding, C.kernel, C.dilation, C.stride)
    Y = reshape(ᵛ(Z), Zchannels, :)
    Xmat = Y * W .+ B
    Xten = mat2ten(Xmat, Zeros(T, xtensize), C.padding, C.kernel, C.dilation, C.stride, C.padmode, C.padval)

    # UNPAD OP HERE
    return C.f(Xten)
end
