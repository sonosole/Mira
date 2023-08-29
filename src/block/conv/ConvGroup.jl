export GroupConv
export GroupConv1d
export GroupConv2d
export GroupConv3d
export GroupConv4d
export GroupConv5d


"""
Applies a D-dims group convolution over an (D+2)-dims input tensor of shape (ichannels, `w1`,`w2`,...,`wD`, batchsize)\n

# Constructor
    GroupConv{D}(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                 groups   :: Int = 2,
                 kernel   :: Dims{D} = ntuple(i -> 3, D),
                 dilation :: Dims{D} = ntuple(i -> 1, D),
                 stride   :: Dims{D} = ntuple(i -> 1, D),
                 padval   :: Real = 0f0,
                 padmode  :: String = "repeat",
                 padding  :: Dims2OrStr = "valid",
                 type     :: Type = Array{Float32}) where D

+ `padmode` should be one of "zeros", "constant", "repeat", "reflect", "symmetric", "circular"
+ `padding` can be "valid", "same", or type `NTuple{D, Dims{2}}`
"""
mutable struct GroupConv{D} <: Block
    blocks :: Vector{Conv{D}}
    groups :: Int
    function GroupConv{D}(ichannels :: Int, ochannels::Int, fn::FunOrNil=relu;
                           groups   :: Int = 2,
                           kernel   :: Dims{D} = ntuple(i -> 3, D),
                           dilation :: Dims{D} = ntuple(i -> 1, D),
                           stride   :: Dims{D} = ntuple(i -> 1, D),
                           padval   :: Real = 0f0,
                           padmode  :: String = "repeat",
                           padding  :: PadsDOrStr = "valid",
                           type     :: Type = Array{Float32}) where D

        if groups < 2
            error("since groups=$groups, use Conv1d instead")
        end
        if mod(ichannels, groups) ≠ 0
            error("input channels must be divisible by groups, but got $ichannels÷$groups")
        end
        if mod(ochannels, groups) ≠ 0
            error("output channels must be divisible by groups, but got $ochannels÷$groups")
        end

        blocks = Vector{Conv{D}}(undef, groups)
        for i in 1:groups
            blocks[i] = Conv{D}(ichannels ÷ groups,
                                ochannels ÷ groups, fn;
                                kernel, dilation, stride,
                                padval, padmode, padding,
                                type)
        end
        new{D}(blocks, groups)
    end
    function GroupConv{D}(groups::Int=2) where D
        new{D}(Vector{Conv{D}}(undef, groups), groups)
    end
end

@inline groups(G::GroupConv) = G.groups
@inline Base.first(G::GroupConv)  = G.blocks[1]
@inline Base.getindex(G::GroupConv, i::Int) = G.blocks[i]

@inline Base.lastindex(G::GroupConv) = G.groups
@inline Base.firstindex(G::GroupConv) = 1

function Base.iterate(G::GroupConv, i::Int=1)
    if i ≤ G.groups
        return G[i], i+1
    end
    return nothing
end

function clone(this::GroupConv{D}; type::Type=Array{Float32}) where D
    cloned = GroupConv{D}(groups(this))
    for i in 1:groups(this)
        cloned.blocks[i] = clone(this.blocks[i], type=type)
    end
    return cloned
end

# pretty show
function Base.show(io::IO, G::GroupConv{1})
    m = first(G)
    g = groups(G)
    P = ifelse(paddings(m.padding)==0, "", " padding=$(first(m.padding)),")
    D = ifelse(first(m.dilation)==1,   "", " dilation=$(first(m.dilation)),")
    S = ifelse(first(m.stride)==1,     "", " stride=$(first(m.stride)),")
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    och  = g * SIZE[2] ÷ prod(m.kernel)
    ich  = g * SIZE[1]
    print(io, "GroupConv1d($och => $ich, $(m.f), groups=$g, kernel=$(first(m.kernel)),$D$S$P type=$TYPE)")
end

function Base.show(io::IO, G::GroupConv{N}) where N
    m = first(G)
    g = groups(G)
    P = ifelse(paddings(m.padding)==0, "", " padding=$(m.padding),")
    D = ifelse(prod(m.dilation)==1,   "", " dilation=$(m.dilation),")
    S = ifelse(prod(m.stride)==1,     "", " stride=$(m.stride),")
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    och  = g * SIZE[2] ÷ prod(m.kernel)
    ich  = g * SIZE[1]
    print(io, "GroupConv$(N)d($och => $ich, $(m.f), groups=$g, kernel=$(m.kernel),$D$S$P type=$TYPE)")
end


function paramsof(G::GroupConv)
    params = Vector{Variable}(undef, 0)
    for conv in G
        append!(params, paramsof(conv))
    end
    return params
end


function xparamsof(G::GroupConv)
    xparams = Vector{XVariable}(undef, 0)
    for conv in G
        append!(xparams, xparamsof(conv))
    end
    return xparams
end


function nparamsof(G::GroupConv)
    return nparamsof(first(G)) * groups(G)
end

elsizeof(G::GroupConv) = elsizeof(first(G))

function bytesof(G::GroupConv, unit::String="MB")
    return bytesof(first(G), unit) * groups(G)
end



function forward(G::GroupConv, x::Variable{T}) where T
    N  = groups(G)
    ys = Vector{Variable{T}}(undef, N)
    xs = divchannel(x, N)
    for i in 1:N
        ys[i] = forward(G[i], xs[i])
    end
    zs = catchannel(ys)
    return zs
end



function predict(G::GroupConv, x::AbstractArray)
    N  = groups(G)
    ys = Vector{AbstractArray}(undef, N)
    xs = divchannel(x, N)
    for i in 1:N
        ys[i] = predict(G[i], xs[i])
    end
    zs = catchannel(ys)
    return zs
end






"""
Applies a `1`-D group convolution over an `3`-D input tensor of shape (ichannels, `steps`, batchsize)\n

# Constructor
    GroupConv1d(ichannels :: Int, ochannels::Int, fn::FunOrNil=relu;
                 groups   :: Int = 2,
                 kernel   :: Int = 3,
                 dilation :: Int = 1,
                 stride   :: Int = 1,
                 padval   :: Real = 0f0,
                 padmode  :: String = "repeat",
                 padding  :: Pads1OrStr = "valid",
                 type     :: Type = Array{Float32})

+ `padmode` should be one of "zeros", "constant", "repeat", "reflect", "symmetric", "circular"
+ `padding` can be "valid", "same", or type `Dims{2}`
"""
function GroupConv1d(ichannels :: Int, ochannels::Int, fn::FunOrNil=relu;
                      groups   :: Int = 2,
                      kernel   :: Int = 3,
                      dilation :: Int = 1,
                      stride   :: Int = 1,
                      padval   :: Real = 0f0,
                      padmode  :: String = "repeat",
                      padding  :: Pads1OrStr = "valid",
                      type     :: Type = Array{Float32})

    if !isa(padding, String)
        padding = singletuple(padding)
    end

    kernel   = singletuple(kernel)
    dilation = singletuple(dilation)
    stride   = singletuple(stride)

    return GroupConv{1}(ichannels, ochannels, fn;
                        groups,
                        kernel, dilation, stride,
                        padval, padmode, padding,
                        type)
end


"""
Applies a `2`-D group convolution over an `4`-D input tensor of shape (ichannels, `hight`, `width`, batchsize)\n

# Constructor
    GroupConv2d(ichannels :: Int, ochannels::Int, fn::FunOrNil=relu;
                 groups   :: Int = 2,
                 kernel   :: Dims{2} = (3,3),
                 dilation :: Dims{2} = (1,1),
                 stride   :: Dims{2} = (1,1),
                 padval   :: Real = 0f0,
                 padmode  :: String = "repeat",
                 padding  :: Pads2OrStr = "valid",
                 type     :: Type = Array{Float32})

+ `padmode` should be one of "zeros", "constant", "repeat", "reflect", "symmetric", "circular"
+ `padding` can be "valid", "same", or type `NTuple{2, Dims{2}}`
"""
function GroupConv2d(ichannels :: Int, ochannels::Int, fn::FunOrNil=relu;
                      groups   :: Int = 2,
                      kernel   :: Dims{2} = (3,3),
                      dilation :: Dims{2} = (1,1),
                      stride   :: Dims{2} = (1,1),
                      padval   :: Real = 0f0,
                      padmode  :: String = "repeat",
                      padding  :: Pads2OrStr = "valid",
                      type     :: Type = Array{Float32})

    return GroupConv{2}(ichannels, ochannels, fn;
                        groups,
                        kernel, dilation, stride,
                        padval, padmode, padding,
                        type)
end



"""
Applies a `3`-D group convolution over an `5`-D input tensor of shape (ichannels, `hight`, `width`, `steps`, batchsize)\n

# Constructor
    GroupConv3d(ichannels :: Int, ochannels::Int, fn::FunOrNil=relu;
                 groups   :: Int     = 2,
                 kernel   :: Dims{3} = (3,3,3),
                 dilation :: Dims{3} = (1,1,1),
                 stride   :: Dims{3} = (1,1,1),
                 padval   :: Real = 0f0,
                 padmode  :: String = "repeat",
                 padding  :: Pads3OrStr = "valid",
                 type     :: Type = Array{Float32})

+ `padmode` should be one of "zeros", "constant", "repeat", "reflect", "symmetric", "circular"
+ `padding` can be "valid", "same", or type `NTuple{3, Dims{2}}`
"""
function GroupConv3d(ichannels :: Int, ochannels::Int, fn::FunOrNil=relu;
                      groups   :: Int     = 2,
                      kernel   :: Dims{3} = (3,3,3),
                      dilation :: Dims{3} = (1,1,1),
                      stride   :: Dims{3} = (1,1,1),
                      padval   :: Real = 0f0,
                      padmode  :: String = "repeat",
                      padding  :: Pads3OrStr = "valid",
                      type     :: Type = Array{Float32})

    return GroupConv{3}(ichannels, ochannels, fn;
                        groups,
                        kernel, dilation, stride,
                        padval, padmode, padding,
                        type)
end



"""
Applies a `4`-D group convolution over an `6`-D input tensor of shape (ichannels, `w1`,`w2`,`w3`,`w4`, batchsize)\n

# Constructor
    GroupConv4d(ichannels :: Int, ochannels::Int, fn::FunOrNil=relu;
                 groups   :: Int = 2,
                 kernel   :: Dims{4} = (3,3,3,3),
                 dilation :: Dims{4} = (1,1,1,1),
                 stride   :: Dims{4} = (1,1,1,1),
                 padval   :: Real = 0f0,
                 padmode  :: String = "repeat",
                 padding  :: Pads4OrStr = "valid",
                 type     :: Type = Array{Float32})

+ `padmode` should be one of "zeros", "constant", "repeat", "reflect", "symmetric", "circular"
+ `padding` can be "valid", "same", or type `NTuple{4, Dims{2}}`
"""
function GroupConv4d(ichannels :: Int, ochannels::Int, fn::FunOrNil=relu;
                      groups   :: Int = 2,
                      kernel   :: Dims{4} = (3,3,3,3),
                      dilation :: Dims{4} = (1,1,1,1),
                      stride   :: Dims{4} = (1,1,1,1),
                      padval   :: Real = 0f0,
                      padmode  :: String = "repeat",
                      padding  :: Pads4OrStr = "valid",
                      type     :: Type = Array{Float32})

    return GroupConv{4}(ichannels, ochannels, fn;
                        groups,
                        kernel, dilation, stride,
                        padval, padmode, padding,
                        type)
end


"""
Applies a `5`-D group convolution over an `7`-D input tensor of shape (ichannels, `w1`,`w2`,`w3`,`w4`,`w5`, batchsize)\n

# Constructor
    GroupConv5d(ichannels :: Int, ochannels::Int, fn::FunOrNil=relu;
                 groups   :: Int = 2,
                 kernel   :: Dims{5} = (3,3,3,3,3),
                 dilation :: Dims{5} = (1,1,1,1,1),
                 stride   :: Dims{5} = (1,1,1,1,1),
                 padval   :: Real = 0f0,
                 padmode  :: String = "repeat",
                 padding  :: Pads5OrStr = "valid",
                 type     :: Type = Array{Float32})

+ `padmode` should be one of "zeros", "constant", "repeat", "reflect", "symmetric", "circular"
+ `padding` can be "valid", "same", or type `NTuple{5, Dims{2}}`
"""
function GroupConv5d(ichannels :: Int, ochannels::Int, fn::FunOrNil=relu;
                      groups   :: Int = 2,
                      kernel   :: Dims{5} = (3,3,3,3,3),
                      dilation :: Dims{5} = (1,1,1,1,1),
                      stride   :: Dims{5} = (1,1,1,1,1),
                      padval   :: Real = 0f0,
                      padmode  :: String = "repeat",
                      padding  :: Pads5OrStr = "valid",
                      type     :: Type = Array{Float32})

    return GroupConv{5}(ichannels, ochannels, fn;
                        groups,
                        kernel, dilation, stride,
                        padval, padmode, padding,
                        type)
end
