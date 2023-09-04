export PlainDepthConv1d

mutable struct PlainDepthConv1d <: Block
    w::VarOrNil # input to hidden weights
    b::VarOrNil # bias of hidden units
    f::FunOrNil # active fn or nothing
    k::Int      # kernel size
    s::Int      # stride size
    function PlainDepthConv1d(channels::Int,
                              fn::FunOrNil=relu;
                              kernel::Int=3,
                              stride::Int=1,
                              gain::Real=1.0,
                              type::Type=Array{Float32})
        T = eltype(type)
        A = T(sqrt(2 / kernel)) .* gain
        w = randn(T, channels, kernel) .* A
        b = zeros(T, channels,      1)
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true),
            fn, kernel, stride)
    end
    function PlainDepthConv1d(fn::FunOrNil; kernel::Int=3, stride::Int=1)
        new(nothing, nothing, fn, kernel, stride)
    end
end


function clone(this::PlainDepthConv1d; type::Type=Array{Float32})
    cloned = PlainDepthConv1d(this.f, kernel=this.k, stride=this.s)
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    return cloned
end


function Base.show(io::IO, m::PlainDepthConv1d)
    SIZE = size(m.w)
    TYPE = typeof(m.w.value)
    print(io, "PlainDepthConv1d($(SIZE[1]), $(m.f); kernel=$(m.k), stride=$(m.s), type=$TYPE)")
end


function paramsof(m::PlainDepthConv1d)
    params = Vector{Variable}(undef, 2)
    params[1] = m.w
    params[2] = m.b
    return params
end


function xparamsof(m::PlainDepthConv1d)
    xparams = Vector{XVariable}(undef, 2)
    xparams[1] = ('w', m.w)
    xparams[2] = ('b', m.b)
    return xparams
end


function nparamsof(m::PlainDepthConv1d)
    lw = length(m.w)
    lb = length(m.b)
    return (lw + lb)
end


elsizeof(c::PlainDepthConv1d) = elsizeof(c.w)


function bytesof(block::PlainDepthConv1d, unit::String="MB")
    n = nparamsof(block) * elsizeof(block)
    return blocksize(n, uppercase(unit))
end


function nops(d::PlainDepthConv1d, n::Int=1)
    c, k = size(d.w)        # channels, kernel
    mops = c * k            # mul ops
    aops = c * (k-1) + c    # add ops
    acts = c                # active ops
    return (mops, aops, acts) .* n
end


function forward(block::PlainDepthConv1d, x::Variable{T}) where T
    @assert ndims(x)==3 "input shape is of (ichannels, width, batchsize)"
    channels, width, batchsize = size(x)
    kernel = block.k
    stride = block.s
    steps = floor(Int, (width-kernel)/stride) + 1
    vy = Zeros(T, channels, steps, batchsize)
    vx = value(x)
    F = block.f
    w = block.w
    b = block.b
    s =      1:stride:width # start indices
    f = kernel:stride:width # final indices
    Threads.@threads for t = 1:steps
        vy[:, t:t, :] = sum(vx[:, s[t]:f[t], :] .* ᵛ(w), dims=2)
        #  C × 1 × B              C × K × B        C × K
    end

    y = Variable{T}(vy, x.backprop)

    if y.backprop
        y.backward = function ∇depthConv1d()
            if needgrad(x)
                zerodelta(x)
                δx = δ(x)
                δy = δ(y)
                if kernel > stride
                    for t = 1:steps
                        δx[:, s[t]:f[t], :] .+= δy[:, t:t, :] .* ᵛ(w)
                        #     C × K × B            C × 1 × B     C × K
                    end
                else
                    Threads.@threads for t = 1:steps
                        δx[:, s[t]:f[t], :] .+= δy[:, t:t, :] .* ᵛ(w)
                    end
                end
            end
            if needgrad(w)
                zerodelta(w)
                δw = δ(w)
                δy = δ(y)
                for t = 1:steps
                    u = sum(δy[:, t:t, :] .* vx[:, s[t]:f[t], :], dims=3)
                    #          C × 1 × B           C × K × B
                    δw .+= reshape(u, channels, kernel)
                end
            end
        end
        addchild(y, w)
        addchild(y, x)
    end

    z = y .+ b
    return F(z)
end


function predict(block::PlainDepthConv1d, x::S) where S <: AbstractArray
    @assert ndims(x)==3 "input shape is of (ichannels, width, batchsize)"
    channels, width, batchsize = size(x)
    kernel = block.k
    stride = block.s
    T = floor(Int, (width-kernel)/stride) + 1
    y = Zeros(S, channels, T, batchsize)
    w = ᵛ(block.w)
    b = ᵛ(block.b)
    F = block.f
    s =      1:stride:width # start indices
    f = kernel:stride:width # final indices
    Threads.@threads for t = 1:T
        y[:, t:t, :] = sum(w .* x[:, s[t]:f[t], :], dims=2)
    end

    z = y .+ b
    return F(z)
end
