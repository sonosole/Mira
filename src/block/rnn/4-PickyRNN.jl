mutable struct PickyRNN <: Block
    w::VarOrNil # input to hidden weights
    b::VarOrNil # bias of hidden units
    f::Function # activation function
    h::Hidden
    function PickyRNN(isize::Int, hsize::Int, fn::Function=relu; type::Type=Array{Float32})
        T = eltype(type)
        a = T(sqrt(2/isize))
        w = randn(T, hsize, isize) .* a
        b = randn(T, hsize,     1) .* a
        new(Variable{type}(w,true,true,true),
            Variable{type}(b,true,true,true),
            fn, nothing)
    end
    function PickyRNN(fn::Function)
        new(nothing, nothing, fn, nothing)
    end
end


function clone(this::PickyRNN; type::Type=Array{Float32})
    cloned   = PickyRNN(this.f)
    cloned.w = clone(this.w, type=type)
    cloned.b = clone(this.b, type=type)
    return cloned
end


function Base.show(io::IO, p::PickyRNN)
    SIZE = size(p.w)
    TYPE = typeof(p.w.value)
    print(io, "PickyRNN($(SIZE[2]), $(SIZE[1]), $(p.f); type=$TYPE)")
end


function resethidden(p::PickyRNN)
    p.h = nothing
end


function paramsof(p::PickyRNN)
    params = Vector{Variable}(undef,2)
    params[1] = p.w
    params[2] = p.b
    return params
end


function xparamsof(p::PickyRNN)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', p.w)
    xparams[2] = ('b', p.b)
    return xparams
end


function nparamsof(p::PickyRNN)
    lw = length(p.w)
    lb = length(p.b)
    return (lw + lb)
end


elsizeof(p::PickyRNN) = elsizeof(p.w)


function bytesof(p::PickyRNN, unit::String="MB")
    n = nparamsof(p) * elsizeof(p)
    return blocksize(n, uppercase(unit))
end


function nops(p::PickyRNN, c::Int=1)
    m, n = size(p.w)
    mops = m * n
    aops = m * (n-1) + m
    acts = m
    return (mops, aops, acts) .* c
end


function forward(p::PickyRNN, x::Variable{T}) where T
    f = p.f  # activition function
    w = p.w  # input's weights
    b = p.b  # input's bias
    C = size(w,1) # feat dims
    B = size(x,2) # batch size
    l = eltype(T)(1.0f0)
    θ = eltype(T)(sqrt(l/ndims(b)))

    z = w * x .+ b                                               # new info
    h = !isnothing(p.h) ? p.h : Variable(Zeros(T, C, B), type=T) # old info
    σ = sigmoid(sum(h .* z, dims=1) .* θ)
    γ = l .- σ
    y   = f(h + σ .* z)
    p.h =   h + γ .* z
    return y
end


function predict(p::PickyRNN, x::T) where T
    f = p.f     # activition function
    w = ᵛ(p.w)  # input's weights
    b = ᵛ(p.b)  # input's bias
    C = size(w,1)
    B = size(x,2)
    l = eltype(T)(1.0f0)
    θ = eltype(T)(sqrt(l/ndims(b)))

    z = w * x .+ b                             # new info
    h = !isnothing(p.h) ? p.h : Zeros(T, C, B) # old info
    σ = sigmoid(sum(h .* z, dims=1) .* θ)      # corr of old-info and new-info
    γ = l .- σ

    y   = f(h + σ .* z)
    p.h =   h + γ .* z
    return y
end
