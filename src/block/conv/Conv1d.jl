export Conv1d
export Conv1dField



"""
    mutable struct Conv1d <: Block

Applies a 1-D convolution over an 3-D input tensor of shape (ichannels, steps, batchsize)\n
"""
mutable struct Conv1d <: Block
    w :: VarOrNil
    b :: VarOrNil
    f :: FunOrNil
    kernel   :: Dims{1}
    dilation :: Dims{1}
    stride   :: Dims{1}
    padding  :: Pads{1}
    padmode  :: Function
    padval   :: Float32
    function Conv1d(ichannels::Int, ochannels::Int, fn::FunOrNil=relu;
                    kernel   :: Int = 3,
                    dilation :: Int = 1,
                    stride   :: Int = 1,
                    padval   :: Real = 0f0,
                    padmode  :: String = "repeat"
                    padding  :: Dims2OrStr = "valid",
                    type     :: Type = Array{Float32})

        if padding isa String
            padding = inferpadding(padding, stride, dialation)
        else
            padding  = (padding,  )
        end

        kernel   = (kernel,   )
        dilation = (dilation, )
        stride   = (stride,   )
        dtype    = eltype(type)

        patchlen = prod(kernel) * ichannels
        Amplifer = dtype(sqrt(2 / patchlen))
        w = Amplifer * randn(dtype, ochannels, patchlen)
        b = Amplifer * randn(dtype, ochannels,        1)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true), fn,
            kernel,
            dilation,
            stride,
            padding,
            selectpad(padmode), padval)
    end
    function Conv1d()
        new(nothing, nothing, nothing, 3, 1, 1, ((0,0),), padzeros, 0f0)
    end
end


function clone(this::Conv1d; type::Type=Array{Float32})
    cloned   = Conv1d()
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    cloned.f = this.f

    cloned.kernel   = this.kernel
    cloned.dilation = this.dilation
    cloned.stride   = this.stride
    cloned.padding  = this.padding
    cloned.padmode  = this.padmode
    cloned.padval   = this.padval
    return cloned
end


# pretty show
function Base.show(io::IO, m::Conv1d)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "Conv1d($(Int(SIZE[2]/m.k)), $(SIZE[1]), $(m.f),
    kernel=$(m.k),
    dilation=$(m.dilation),
    stride=$(m.s),
    padding=$(m.padding),
    padmode=$(m.padmode); type=$TYPE)")
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



function forward(C::Conv1d, x::Variable)
    w = C.w
    b = C.b

    y = im2col(x, C.padding, C.kernel, C.dilation, C.stride, C.padval)
    z = matAddVec(w * y, b)
    z = col2im(z, spatialdims(z,))
    return C.f(z)
end


function predict(C::Conv1d, x::AbstractArray)
    w = value(C.w)
    b = value(C.b)

    y = im2col(x, C.padding, C.kernel, C.dilation, C.stride, C.padval)
    z = col2im(w * y .+ b, zsize)
    return C.f(z)
end
