export TransConv
export TransConv1d
export TransConv2d
export TransConv3d
export TransConv4d
export TransConv5d


"""
Applies a `D`-dim convolution over an `(D+2)`-dim input tensor of shape (ichannels, w1, w2, ..., wn, batchsize)\n
# Constructor
    TransConv{D}(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                 kernel   :: Dims{D} = ntuple(i -> 3, D),
                 dilation :: Dims{D} = ntuple(i -> 1, D),
                 stride   :: Dims{D} = ntuple(i -> 1, D),
                 padding  :: PadsDOrStr = "valid",
                 type     :: Type = Array{Float32}) where D

+ `padmode` should be one of "zeros", "constant", "repeat", "reflect", "symmetric", "circular"
+ `padding` can be "valid", "same", or type `NTuple{D, Dims{2}}`
# Detailed Processes
+ Ordinary Conv Processes:
    `X` → [padfn] → `Xten` → [ten2mat] → `Xmat` → [W*(∙) + B] → `Y` → [reshape] → `Z`
+ Transpose Conv Processes:
    `X` ← [unpad] ← `Xten` ← [mat2ten] ← `Xmat` ← [W*(∙) + B] ← `Y` ← [reshape] ← `Z`
"""
mutable struct TransConv{D} <: Block
    w :: VarOrNil
    b :: VarOrNil
    f :: FunOrNil
    kernel   :: Dims{D}
    dilation :: Dims{D}
    stride   :: Dims{D}
    padding  :: Pads{D}
    outshape :: DimsOrNil
    function TransConv{D}(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                          kernel   :: Dims{D} = ntuple(i -> 3, D),
                          dilation :: Dims{D} = ntuple(i -> 1, D),
                          stride   :: Dims{D} = ntuple(i -> 1, D),
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
        patchlen = prod(kernel) * ochannels
        Amplifer = dtype(sqrt(2 / patchlen))

        w = Amplifer * randn(dtype, patchlen, ichannels)
        b = Amplifer * randn(dtype, patchlen,        1)

        new{D}(Variable{type}(w,true,true,true),
               Variable{type}(b,true,true,true), fn,
               kernel, dilation, stride, npads, nothing)
    end
    function TransConv{D}() where D
        O = (0, 0)
        new{D}(nothing, nothing, nothing,
               ntuple(i -> 3, D),
               ntuple(i -> 1, D),
               ntuple(i -> 1, D),
               ntuple(i -> O, D), nothing)
    end
end


function Base.show(io::IO, c::TransConv{N}) where N
    P = ifelse(paddings(c.padding)==0, "", " padding=$(c.padding),")
    D = ifelse(prod(c.dilation)==1,   "", " dilation=$(c.dilation),")
    S = ifelse(prod(c.stride)==1,     "", " stride=$(c.stride),")
    SIZE =   size(c.w.value)
    TYPE = typeof(c.w.value)
    och  = SIZE[1] ÷ prod(c.kernel)
    ich  = SIZE[2]
    print(io, "TransConv$(N)d($ich => $och, $(c.f), kernel=$(c.kernel),$D$S$P type=$TYPE)")
end

function Base.show(io::IO, c::TransConv{1})
    P = ifelse(paddings(c.padding)==0, "", " padding=$(first(c.padding)),")
    D = ifelse(first(c.dilation)==1,   "", " dilation=$(first(c.dilation)),")
    S = ifelse(first(c.stride)==1,     "", " stride=$(first(c.stride)),")
    SIZE =   size(c.w.value)
    TYPE = typeof(c.w.value)
    och  = SIZE[1] ÷ prod(c.kernel)
    ich  = SIZE[2]
    print(io, "TransConv($ich => $och, $(c.f), kernel=$(first(c.kernel)),$D$S$P type=$TYPE)")
end


"""
+ Ordinary Conv Processes:
X → ([padfn] → Xten → [ten2mat] → Xmat → [W*(∙) + B] → Y → [reshape]) → Z

+ Transpose Conv Processes:
X ← ([unpad] ← Xten ← [mat2ten] ← Xmat ← [W*(∙) + B] ← Y ← [reshape]) ← Z
"""
function forward(C::TransConv{D}, Z::Variable{T}) where {T,D}
    W = C.w
    B = C.b
    N = D + 2

    ZSIZE = size(Z)
    YSIZE = (ZSIZE[1], prod(ZSIZE[2:N]))

    Y = reshape(Z, YSIZE)
    X = mat2ten(W * Y .+ B, ZSIZE, C.outshape, C.padding, C.kernel, C.dilation, C.stride)
    return C.f(X)
end


function predict(C::TransConv{D}, Z::AbstractArray) where D
    W = value(C.w)
    B = value(C.b)
    N = D + 2

    ZSIZE = size(Z)
    YSIZE = (ZSIZE[1], prod(ZSIZE[2:N]))

    Y = reshape(Z, YSIZE)
    X = mat2ten(W * Y .+ B, ZSIZE, C.outshape, C.padding, C.kernel, C.dilation, C.stride)
    return C.f(X)
end
