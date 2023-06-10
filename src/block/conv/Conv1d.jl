export Conv1d
export Conv1dField

const IntOrStr = Union{Int, String}

function selectpadfn(padmode::String)
    if padmode == "zeros"
        return padzeros
    elseif padmode == "constant"
        return padconst
    elseif padmode == "repeat"
        return padrepeat
    elseif padmode == "reflect"
        return padreflect
    elseif padmode == "symmetric"
        return padsymmetric
    elseif padmode == "circular"
        return padcircular
    else
        error("padmode should be one of \"zeros\", \"constant\", \"repeat\", \"reflect\", \"symmetric\",\"circular\"")
    end
end

"""
    mutable struct Conv1d <: Block

Applies a 1-D convolution over an 3-D input tensor of shape (ichannels, steps, batchsize)\n
"""
mutable struct Conv1d <: Block
    w :: VarOrNil
    b :: VarOrNil
    f :: FunOrNil
    kernel   :: Int
    dilation :: Int
    stride   :: Int
    padding  :: Int
    padmode  :: Function
    function Conv1d(ichannels::Int,
                    ochannels::Int,
                    fn::FunOrNil=relu;
                    kernel   :: Int = 3,
                    dilation :: Int = 1,
                    stride   :: Int = 1,
                    padding  :: IntOrStr = 0,
                    padmode  :: String   = "repeat"
                    padval   :: Real     = 0.0,
                    type::Type=Array{Float32})

        if isa(padding, String)
            if padding ∉ ("same", "valid")
                error("padmode should be \"zeros\" or \"const\", but got $padding")
            end
            if isequal(padding, "same") && stride≠1
                error("when padding==\"same\", stride should be 1, but got $stride")
            end
        end

        dtype = eltype(type)
        filterSize = ichannels * kernel
        A = dtype(sqrt(2 / filterSize))
        w = A * randn(dtype, ochannels, filterSize)
        b = A * randn(dtype, ochannels,          1)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true), fn,
            kernel,
            dilation,
            stride,
            padding,
            selectpad(padmode))
    end
    function Conv1d(fn::FunOrNil; kernel::Int=3, stride::Int=1)
        new(nothing, nothing, fn, kernel, stride)
    end
end


function clone(this::Conv1d; type::Type=Array{Float32})
    cloned = Conv1d(this.f, kernel=this.k, stride=this.s)
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    return cloned
end


# pretty show
function Base.show(io::IO, m::Conv1d)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "Conv1d($(Int(SIZE[2]/m.k)), $(SIZE[1]), $(m.f), kernel=$(m.k), stride=$(m.s); type=$TYPE)")
end


"""
    unbiasedof(m::Conv1d)

unbiased weights of Conv1d block
"""
function unbiasedof(m::Conv1d)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end


function weightsof(m::Conv1d)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function gradsof(m::Conv1d)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds!(m::Conv1d)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::Conv1d)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::Conv1d)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::Conv1d)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end

elsizeof(c::Conv1d) = elsizeof(c.w)

function bytesof(model::Conv1d, unit::String="MB")
    n = nparamsof(model) * elsizeof(model.w)
    return blocksize(n, uppercase(unit))
end


"""
    Conv1dField(StrideKernelPair::Vector{NTuple{2,Int}})
# Example
    julia> Conv1dField([(3,2),(3,1),(4,2)]
    (1:13, 5:17)
"""
function Conv1dField(StrideKernelPair::Vector{NTuple{2,Int}})
    # 输入是从底层到顶层的(kernel,stride)列表
    # 计算感受野时从顶层往底层计算,为了流式计算时候缓存空间的设计
    # 本函数返回：顶层第一个时间步感受到的底层时间步范围
    #            顶层第二个时间步感受到的底层时间步范围
    t1 = 1
    t2 = 2
    for (kernel,stride) in StrideKernelPair[end:-1:1]
        t1 = (t1-1) * stride + kernel
        t2 = (t2-1) * stride + kernel
    end
    return (1:t1,t2-t1+1:t2)
end


function Conv1dField(chain)
    # 计算感受野时从顶层往底层计算,为了流式计算时候缓存空间的设计
    # 本函数返回：顶层第一个时间步感受到的底层时间步范围
    #            顶层第二个时间步感受到的底层时间步范围
    t1 = 1
    t2 = 2
    for i = length(chain):-1:1
        if typeof(chain[i]) <: Conv1d
            t1 = (t1-1) * chain[i].s + chain[i].k
            t2 = (t2-1) * chain[i].s + chain[i].k
        end
    end
    return (1:t1,t2-t1+1:t2)
end



function forward(block::Conv1d, x::Variable{T}) where T
    w = block.w
    b = block.b
    f = block.f

    x = im2col(x, block.k, block.s)
    x = matAddVec(w * x, b)
    x = col2im(x, batchsize)
    return f(x)
end


function predict(block::Conv1d, x::AbstractArray)
    f = block.f
    w = value(block.w)
    b = value(block.b)

    x = im2col(x, block.k, block.s)
    x = w * x .+ b
    x = col2im(x, batchsize)
    return f(x)
end
