export LayerNorm

"""
# Explanation
Applies mean and variance normalization over a N-dimensional input `x`. μ and σ
are collected from each sample, e.g. one single word in NLP (NOT sequtial of words)
one single picture in CV. Suppose `x` has shape (C, `W1,W2,...,Wd`, B), and reshape
`x` to shape (C, `T`, B).
## Sequential samples
Usually `x` is composed of batched sequential samples (maybe different sequence lengths
but padded to the same length, e.g. speech features or words embeddings). then
+ μ[t,b] = mean(x[1:C, t, b]), t ∈ 1,2,...,T and b ∈ 1,2,...,B
+ σ[t,b] =  std(x[1:C, t, b]), t ∈ 1,2,...,T and b ∈ 1,2,...,B
## None-Sequential samples
If `x` is batched samples without sequential concepts, then
+ μ[b] = mean(x[1:C, 1:T, 1:B]), c ∈ 1,2,...,C
+ σ[b] =  std(x[1:C, 1:T, 1:B]), c ∈ 1,2,...,C
for example an single picture sample has RGB channles
    ┌┬┬┬┬┬┬┬┐R Channel
    ├┼┼┼┼┼┼┼┤
    H┼┼┼┼┼┼┼┤    ────────────────────────┐
    ├┼┼┼┼┼┼┼┤                            │
    └┴┴┴W┴┴┴┘                            │
    ┌┬┬┬┬┬┬┬┐G Channel                   │
    ├┼┼┼┼┼┼┼┤                            ▼
    H┼┼┼┼┼┼┼┤    ───────────────────────►▓ μ or σ (shape of 1×1)
    ├┼┼┼┼┼┼┼┤                            ▲
    └┴┴┴W┴┴┴┘                            │
    ┌┬┬┬┬┬┬┬┐B Channel                   │
    ├┼┼┼┼┼┼┼┤                            │
    H┼┼┼┼┼┼┼┤   ─────────────────────────┘
    ├┼┼┼┼┼┼┼┤
    └┴┴┴W┴┴┴┘
# Constructor
    LayerNorm(;ndims :: Int,
               dims  :: IntOrDims{D},
               size  :: IntOrDims{D},
               eps   :: AbstractFloat=1e-38,
               type  :: Type=Array{Float32}) where D
"""
mutable struct LayerNorm <: Normalizer
    γ :: VarOrNil               # scaling params
    β :: VarOrNil               # shifting params
    dims :: Union{Int,NTuple}   # dims is where μ and σ come from
    eps  :: AbstractFloat       # prevent zero-dividing
    function LayerNorm(;ndims :: Int,
                        dims  :: IntOrDims{D},
                        size  :: IntOrDims{D},
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
        new(γ, β, dims, eps)
    end
    function LayerNorm(dims, eps)
        new(nothing, nothing, dims, eps)
    end
end


function forward(l::LayerNorm, x::Variable{T}) where T
    γ = l.γ
    β = l.β
    x̌ = znorm(x, dims=l.dims, eps=l.eps)
    y = x̌ .* γ .+ β
    return y
end


function predict(l::LayerNorm, x::AbstractArray)
    γ = ᵛ(l.γ)
    β = ᵛ(l.β)
    x̌ = znorm(x, dims=l.dims, eps=l.eps)
    y = x̌ .* γ .+ β
    return y
end


function clone(this::LayerNorm; type::Type=Array{Float32})
    cloned    = LayerNorm(this.dims, this.eps)
    cloned.γ  = clone(this.γ, type=type)
    cloned.β  = clone(this.β, type=type)
    return cloned
end

function Base.show(io::IO, l::LayerNorm)
    SIZE =   size(l.β.value)
    TYPE = typeof(l.β.value)
    DIMS = l.dims
    print(io, "LayerNorm(dims=$DIMS,size=$SIZE; type=$TYPE)")
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
