export BatchNorm

"""
    bn_shape_and_rdims(ndims::Int, dim::Int, chs::Int) -> shape::Dims, reducingdims::Dims

Return the affine vars' shape and reducing dims.
+ `ndims` is the number of dims of `BatchNorm`
+ `dim` is the channel-dim
+ `chs` is the number of channels
"""
function bn_shape_and_rdims(ndims::Int, dim::Int, chs::Int)
    @assert ndims ≥ dim "total dims shall NOT be smaller than channel-dim"
    shape = ntuple(i -> i==dim ? chs : 1, ndims)
    rdims = ntuple(i -> i>=dim ? i+1 : i, ndims-1)
    return shape, rdims
end


"""
# Explanation
Applies mean and variance normalization over a N-dimensional input `x`. Suppose `x` has
shape (C, `W1,W2,...,Wd`, B), and reshape `x` to shape (C, `T`, B), Each element of `μ` and `σ`
are collected from each channle of all samples, i.e.
+ μ[c] = mean(x[c, 1:T, 1:B]), c ∈ 1:C
+ σ[c] =  std(x[c, 1:T, 1:B]), c ∈ 1:C
finally, the shape of `μ` or `σ` is (C, `1,1,...,1`, 1).
# Constructor
    BatchNorm(;ndims    :: Int,                  # how many dimentions the input data has
               dim      :: Int,                  # channle dim
               channels :: Int,                  # channle size
               inertia  :: AbstractFloat=0.9,    # smoothing const or historical inertia
               eps      :: AbstractFloat=1e-38,
               type     :: Type=Array{Float32})

# N-dimentional Constructor
+ BatchNorm0d(channels::Int; inertia=0.9, eps=1e-38, type=Array{Float32}) where D
+ BatchNorm1d(channels::Int; inertia=0.9, eps=1e-38, type=Array{Float32}) where D
+ BatchNorm2d(channels::Int; inertia=0.9, eps=1e-38, type=Array{Float32}) where D
+ BatchNorm3d(channels::Int; inertia=0.9, eps=1e-38, type=Array{Float32}) where D
+ BatchNorm4d(channels::Int; inertia=0.9, eps=1e-38, type=Array{Float32}) where D
+ BatchNorm5d(channels::Int; inertia=0.9, eps=1e-38, type=Array{Float32}) where D
"""
mutable struct BatchNorm <: Normalizer
    γ    :: VarOrNil                    # scaling params
    β    :: VarOrNil                    # shifting params
    μ    :: Union{AbstractArray,Nil}    # running average
    σ²   :: Union{AbstractArray,Nil}    # running variance
    ϵ    :: AbstractFloat               # prevent zero-dividing
    ρ    :: AbstractFloat               # inertia coefficient
    dims :: NTuple                      # dims to collect elements for mean and variance
    function BatchNorm(;ndims    :: Int,                  # data dimentions
                        dim      :: Int,                  # channle dim
                        channels :: Int,                  # channle size
                        inertia  :: AbstractFloat=0.9,    # smoothing const or historical inertia
                        eps      :: AbstractFloat=1e-38,
                        type     :: Type=Array{Float32})

        shape, dims2reduce = bn_shape_and_rdims(ndims, dim, channels)
        γ  = Variable{type}( Ones(type, shape), true, true, true)
        β  = Variable{type}(Zeros(type, shape), true, true, true)
        μ  = Zeros(type, shape)
        σ² =  Ones(type, shape)
        ρ  = eltype(type)(inertia)
        new(γ, β, μ, σ², eps, ρ, dims2reduce)
    end
    function BatchNorm(ϵ::Real, ρ::Real, dims::Union{Tuple,Int})
        new(nothing, nothing, nothing, nothing, ϵ, ρ, dims)
    end
end



function forward(b::BatchNorm, x::Variable)
    γ = b.γ
    β = b.β

    x̌, μ, σ² = znorm_mean_var(x, dims=b.dims, eps=b.ϵ)
    y = x̌ .* γ .+ β

    T = eltype(x)
    l = T(1)
    ρ = T(b.ρ)

    𝝁  = b.μ
    𝝈² = b.σ²
    @sync begin
        @async @. 𝝁  = ρ * 𝝁  + (l - ρ) * μ    # running mean
        @async @. 𝝈² = ρ * 𝝈² + (l - ρ) * σ²   # running var
    end
    return y
end


function predict(b::BatchNorm, x::AbstractArray)
    l  = eltype(x)(1)
    γ  = ᵛ(b.γ)
    β  = ᵛ(b.β)
    μ  = b.μ
    σ² = b.σ²

    x̌ = @. (x - μ) * (l / sqrt(σ²))
    y = @. x̌ * γ + β
    return y
end


function clone(this::BatchNorm; type::Type=Array{Float32})
    cloned    = BatchNorm(this.ϵ, this.ρ, this.dims)
    cloned.γ  = clone(this.γ, type=type)
    cloned.β  = clone(this.β, type=type)
    cloned.μ  = type(this.μ)
    cloned.σ² = type(this.σ²)
    return cloned
end

function Base.show(io::IO, bn::BatchNorm)
    S = length(bn.β.value)
    T = typeof(bn.β.value)
    D = length(bn.dims) - 1
    ρ = bn.ρ
    print(io, "BatchNorm$(D)d(channels=$S, inertia=$ρ; type=$T)")
end

function paramsof(bn::BatchNorm)
    params = Vector{Variable}(undef,2)
    params[1] = bn.γ
    params[2] = bn.β
    return params
end

function xparamsof(bn::BatchNorm)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', bn.γ)
    xparams[2] = ('b', bn.β)
    return xparams
end

function nparamsof(bn::BatchNorm)
    return 4*length(bn.β)
end

elsizeof(bn::BatchNorm) = elsizeof(bn.γ)

function bytesof(bn::BatchNorm, unit::String="MB")
    n = nparamsof(bn) * elsizeof(bn)
    return blocksize(n, uppercase(unit))
end


for DIMS in [0,1,2,3,4,5]
    @eval begin
        export $(Symbol("BatchNorm$(DIMS)d"))
        function $(Symbol("BatchNorm$(DIMS)d"))(channels :: Int;
                                                eps      :: AbstractFloat=1e-38,
                                                inertia  :: AbstractFloat=0.9,
                                                type     :: Type=Array{Float32})
            return BatchNorm(;ndims=$(DIMS+2), dim=1, channels, eps, inertia, type)
        end
    end
end
