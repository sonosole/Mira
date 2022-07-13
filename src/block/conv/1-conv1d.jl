export PlainConv1d
export PlainConv1dReceptiveField


"""
    mutable struct PlainConv1d <: Block

Applies a 1-D convolution over an 3-D input tensors.\n
    Input 3D-tensor of shape (ichannels, timeSteps, batchsize)\n
    Filter 3D-tensor of shape (ochannels, ichannels, kernel)\n
actually the Filter is reshaped to a 2D-tensor of shape (ochannels, ichannels*kernel)
for convenient. This is the simplest case which has just kernel and stride parameters.
"""
mutable struct PlainConv1d <: Block
    w::VarOrNil # input to hidden weights
    b::VarOrNil # bias of hidden units
    f::FunOrNil # active fn or nothing
    k::Int      # kernel size
    s::Int      # stride size
    function PlainConv1d(ichannels::Int,
                         ochannels::Int,
                         fn::FunOrNil=relu;
                         kernel::Int=3,
                         stride::Int=1,
                         type::Type=Array{Float32})

        dtype = eltype(type)
        filterSize = ichannels * kernel
        A = dtype(sqrt(2 / filterSize))
        w = A * randn(dtype, ochannels, filterSize)
        b = A * randn(dtype, ochannels,          1)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true),
            fn, kernel, stride)
    end
    function PlainConv1d(fn::FunOrNil; kernel::Int=3, stride::Int=1)
        new(nothing, nothing, fn, kernel, stride)
    end
end


function clone(this::PlainConv1d; type::Type=Array{Float32})
    cloned = PlainConv1d(this.f, kernel=this.k, stride=this.s)
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    return cloned
end


# pretty show
function Base.show(io::IO, m::PlainConv1d)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "PlainConv1d($(Int(SIZE[2]/m.k)), $(SIZE[1]), $(m.f), kernel=$(m.k), stride=$(m.s); type=$TYPE)")
end


"""
    unbiasedof(m::PlainConv1d)

unbiased weights of PlainConv1d block
"""
function unbiasedof(m::PlainConv1d)
    weights = Vector(undef, 1)
    weights[1] = m.w.value
    return weights
end


function weightsof(m::PlainConv1d)
    weights = Vector(undef, 2)
    weights[1] = m.w.value
    weights[2] = m.b.value
    return weights
end


function gradsof(m::PlainConv1d)
    grads = Vector(undef, 2)
    grads[1] = m.w.delta
    grads[2] = m.b.delta
    return grads
end


function zerograds!(m::PlainConv1d)
    for v in gradsof(m)
        v .= 0.0
    end
end


function paramsof(m::PlainConv1d)
    params = Vector{Variable}(undef,2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::PlainConv1d)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::PlainConv1d)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end

elsizeof(c::PlainConv1d) = elsizeof(c.w)

function bytesof(model::PlainConv1d, unit::String="MB")
    n = nparamsof(model) * elsizeof(model.w)
    return blocksize(n, uppercase(unit))
end


"""
    PlainConv1dReceptiveField(StrideKernelPair::Vector{NTuple{2,Int}})
# Example
    julia> PlainConv1dReceptiveField([(3,2),(3,1),(4,2)]
    (1:13, 5:17)
"""
function PlainConv1dReceptiveField(StrideKernelPair::Vector{NTuple{2,Int}})
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


function PlainConv1dReceptiveField(chain::Chain)
    # 计算感受野时从顶层往底层计算,为了流式计算时候缓存空间的设计
    # 本函数返回：顶层第一个时间步感受到的底层时间步范围
    #            顶层第二个时间步感受到的底层时间步范围
    t1 = 1
    t2 = 2
    for i = length(chain):-1:1
        if typeof(chain[i]) <: PlainConv1d
            t1 = (t1-1) * chain[i].s + chain[i].k
            t2 = (t2-1) * chain[i].s + chain[i].k
        end
    end
    return (1:t1,t2-t1+1:t2)
end


# in2col for predict
function in2col(var::Array{T}, kernel::Int, stride::Int) where T
    # from (ichannels,width,batchsize) to (ichannels*kernel,cols)
    (ichannels,width,batchsize) = size(var)
    step = floor(Int,(width-kernel)/stride + 1)
    cols = step * batchsize
    rows = ichannels * kernel
    out  = zeros(T, rows, cols)
    Threads.@threads for b = 1:batchsize
        index = 1 + (b-1)*step
        start = 1
        final = kernel
        for s = 1:step
            out[:,index] = reshape(var[:,start:final,b], (rows,1))
            start += stride
            final += stride
            index += 1
        end
    end
    return out
end

# in2col for training
function in2col(x::Variable{Array{T}}, kernel::Int, stride::Int) where T
    # x from (ichannels,width,batchsize) to (ichannels*kernel,cols)
    # in which cols = (width – kernel + 1) * batchsize
    (ichannels,width,batchsize) = size(x)
    step = floor(Int,(width-kernel)/stride + 1)
    cols = step * batchsize
    rows = ichannels * kernel
    y = Variable{Array{T}}(zeros(T, rows, cols), x.backprop)

    Threads.@threads for b = 1:batchsize
        index = 1 + (b-1)*step
        start = 1
        final = kernel
        for s = 1:step
            y.value[:,index] = reshape(x.value[:,start:final,b], (rows,1))
            start += stride
            final += stride
            index += 1
        end
    end

    if y.backprop
        y.backward = function ∇in2col()
            if need2computeδ!(x)
                Threads.@threads for b = 1:batchsize
                    index = 1 + (b-1)*step
                    start = 1
                    final = kernel
                    for s = 1:step
                        x.delta[:,start:final,b] += reshape(y.delta[:,index], (ichannels, kernel))
                        start += stride
                        final += stride
                        index += 1
                    end
                end
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end

# col2out for predict
function col2out(x::AbstractArray, batchsize::Int)
    # from (ochannels,width*batchsize) to (ochannels,width,batchsize)
    (ochannels, cols) = size(x)
    width = div(cols, batchsize)
    return reshape(x, (ochannels, width, batchsize))
end

# col2out for training
function col2out(x::Variable, batchsize::Int)
    # from (ochannels,width*batchsize) to (ochannels,width,batchsize)
    (ochannels, cols) = size(x)
    width = div(cols, batchsize)
    return reshape(x, (ochannels, width, batchsize))
end


function forward(block::PlainConv1d, x::Variable{T}) where T
    # size(x) == (ichannels,width,batchsize)
    @assert ndims(x)==3 "input shape is of (ichannels,width,batchsize)"
    batchsize = size(x,3)
    w = block.w
    b = block.b
    f = block.f
    x = in2col(x, block.k, block.s)
    x = matAddVec(w * x, b)
    x = col2out(x, batchsize)
    return f(x)
end


function predict(block::PlainConv1d, x::AbstractArray)
    # size(x) == (ichannels,width,batchsize)
    @assert ndims(x)==3 "input shape is of (ichannels,width,batchsize)"
    batchsize = size(x,3)
    f = block.f
    w = block.w.value
    b = block.b.value
    x = in2col(x, block.k, block.s)
    x = w * x .+ b
    x = col2out(x, batchsize)
    return f(x)
end


function to(type::Type, m::PlainConv1d)
    m.w = to(type, m.w)
    m.b = to(type, m.b)
    return m
end


function to!(type::Type, m::PlainConv1d)
    m.w = to(type, m.w)
    m.b = to(type, m.b)
    return nothing
end
