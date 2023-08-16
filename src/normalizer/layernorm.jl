export LayerNorm

"""
# Explanation
Applies mean and variance normalization over a N-dimensional input `x`. μ and σ
are collected from each sample, e.g. one single word in NLP (NOT sequtial of words)
one single picture in CV. Suppose `x` has shape (C, `W1,W2,...,Wd`, B), and reshape
`x` to shape (C, `T`, B).
## Sequential Samples
Usually `x` is composed of batched sequential samples (maybe different sequence lengths
but padded to the same length, e.g. speech features or words embeddings). then
+ μ[t,b] = mean(x[1:C, t, b]), t ∈ 1:T and b ∈ 1:B
+ σ[t,b] =  std(x[1:C, t, b]), t ∈ 1:T and b ∈ 1:B
## None-Sequential Samples
If `x` is batched samples without sequential concepts, then
+ μ[b] = mean(x[1:C, 1:T, b]), b ∈ 1:B
+ σ[b] =  std(x[1:C, 1:T, b]), b ∈ 1:B
### A Picture Sample Has RGB Channles
    ┌┬┬┬┬┬┬┬┐R Channel
    ├┼┼┼┼┼┼┼┤
    H┼┼┼┼┼┼┼┤    ────────────────────────┐
    ├┼┼┼┼┼┼┼┤                            │
    └┴┴┴W┴┴┴┘                            │
    ┌┬┬┬┬┬┬┬┐G Channel                   ▼
    ├┼┼┼┼┼┼┼┤
    H┼┼┼┼┼┼┼┤    ──────────────────────► ▓ μ or σ (shape of 1×1)
    ├┼┼┼┼┼┼┼┤
    └┴┴┴W┴┴┴┘                            ▲
    ┌┬┬┬┬┬┬┬┐B Channel                   │
    ├┼┼┼┼┼┼┼┤                            │
    H┼┼┼┼┼┼┼┤    ────────────────────────┘
    ├┼┼┼┼┼┼┼┤
    └┴┴┴W┴┴┴┘
all elements from RGB channles are used to estimate mean and variance.

# Constructor
    LayerNorm(;ndims :: Int,                            # data dimentions
               dims  :: IntOrDims{D},                   # dims to reduce mean and variance
               size  :: IntOrDims{D},                   # size of the corresponding dims arg
               eps   :: AbstractFloat=1e-38,            # avoid zero-dividing
               type  :: Type=Array{Float32}) where D

the specified reducing `dims` has shape `size`, e.g. if dims=(3,1), size=(W,C), then the
1-st reducing dim has width C, the 3-rd reducing dim has width W. `ndims` is data dimentions.

# Example
```julia
# NLP Example
ichannels, timesteps, batchsize = 128, 32, 8;
embedding = Variable(randn(Float32, ichannels, timesteps, batchsize));
layernorm = LayerNorm(ndims=3, dims=1, size=ichannels)
y = forward(layernorm, embedding)

# CV Example
C,H,W,B = 3, 256,256, 32
image = Variable(randn(Float32, C,H,W,B));
layernorm = LayerNorm(ndims=4, dims=(3,2,1), size=(W,H,C))
y = forward(layernorm, image)
```

# N-dimentional Constructor
+ LayerNorm0d(;dims::IntOrDims{D}, size::IntOrDims{D}, eps=1e-38, type=Array{Float32}) where D
+ LayerNorm1d(;dims::IntOrDims{D}, size::IntOrDims{D}, eps=1e-38, type=Array{Float32}) where D
+ LayerNorm2d(;dims::IntOrDims{D}, size::IntOrDims{D}, eps=1e-38, type=Array{Float32}) where D
+ LayerNorm3d(;dims::IntOrDims{D}, size::IntOrDims{D}, eps=1e-38, type=Array{Float32}) where D
+ LayerNorm4d(;dims::IntOrDims{D}, size::IntOrDims{D}, eps=1e-38, type=Array{Float32}) where D
+ LayerNorm5d(;dims::IntOrDims{D}, size::IntOrDims{D}, eps=1e-38, type=Array{Float32}) where D
"""
mutable struct LayerNorm <: Normalizer
    γ    :: VarOrNil             # scaling params
    β    :: VarOrNil             # shifting params
    ϵ    :: AbstractFloat        # prevent zero-dividing
    dims :: Union{Int,NTuple}    # dims is where μ and σ come from
    function LayerNorm(;ndims :: Int,           # data dimentions
                        dims  :: IntOrDims{D},  # dims to reduce mean and variance
                        size  :: IntOrDims{D},  # size of the corresponding dims arg
                        eps   :: AbstractFloat=1e-38,
                        type  :: Type=Array{Float32}) where D
        if typeof(dims) ≠ typeof(size)
            error("dims=$dims, size=$size, but they shall be the same type and length")
        end

        shape = ones(Int, ndims)
        for (i, d) in enumerate(dims)
            shape[d] = size[i]
        end
        sz = ntuple(i -> shape[i], ndims)
        γ  = Variable{type}( Ones(type, sz), true, true, true)
        β  = Variable{type}(Zeros(type, sz), true, true, true)
        new(γ, β, eps, dims)
    end
    function LayerNorm(ϵ::Real, dims::IntOrDims{D}) where D
        new(nothing, nothing, ϵ, dims)
    end
end


function forward(l::LayerNorm, x::Variable)
    γ = l.γ
    β = l.β
    x̌ = znorm(x, dims=l.dims, eps=l.ϵ)
    y = x̌ .* γ .+ β
    return y
end


function predict(l::LayerNorm, x::AbstractArray)
    γ = ᵛ(l.γ)
    β = ᵛ(l.β)
    x̌ = znorm(x, dims=l.dims, eps=l.ϵ)
    y = x̌ .* γ .+ β
    return y
end


function clone(this::LayerNorm; type::Type=Array{Float32})
    cloned   = LayerNorm(this.ϵ, this.dims)
    cloned.γ = clone(this.γ, type=type)
    cloned.β = clone(this.β, type=type)
    return cloned
end

function Base.show(io::IO, l::LayerNorm)
    S =   size(l.β.value)
    T = typeof(l.β.value)
    N = ndims(l.β.value) - 2
    D = l.dims
    print(io, "LayerNorm$(N)d(dims=$D, affinesize=$S; type=$T)")
end

function paramsof(l::LayerNorm)
    params = Vector{Variable}(undef,2)
    params[1] = l.γ
    params[2] = l.β
    return params
end

function xparamsof(l::LayerNorm)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', l.γ)
    xparams[2] = ('b', l.β)
    return xparams
end

function nparamsof(l::LayerNorm)
    return 2*length(l.β)
end

elsizeof(l::LayerNorm) = elsizeof(l.γ)

function bytesof(l::LayerNorm, unit::String="MB")
    n = nparamsof(l) * elsizeof(l)
    return blocksize(n, uppercase(unit))
end


for DIMS in [0,1,2,3,4,5]
    @eval begin
        export $(Symbol("LayerNorm$(DIMS)d"))
        function $(Symbol("LayerNorm$(DIMS)d"))(;dims :: IntOrDims{D},
                                                 size :: IntOrDims{D},
                                                 eps  :: AbstractFloat=1e-38,
                                                 type :: Type=Array{Float32}) where D
            return LayerNorm(;ndims=$(2+DIMS), dims, size, eps, type)
        end
    end
end
