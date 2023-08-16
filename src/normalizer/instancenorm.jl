export InstanceNorm

"""
    in_shape_and_rdims(ndims::Int, cdim::Int, bdim::Int, chs::Int) -> shape::Dims, reducingdims::Dims

Return the affine vars' shape and reducing dims.
+ `ndims` is the number of dims of `InstanceNorm`
+ `cdim` is the channel-dim
+ `bdim` is the batch-dim
+ `chs` is the number of channels
"""
function in_shape_and_rdims(ndims::Int, cdim::Int, bdim::Int, chs::Int)
    @assert ndims ≥ cdim "total dims shall NOT be smaller than channel-dim"
    @assert ndims ≥ bdim "total dims shall NOT be smaller than batch-dim"
    @assert cdim  ≠ bdim "channel-dim ≠ batch-dim shall be met"
    shape = ntuple(i -> i==cdim ? chs : 1, ndims)
    dvec  = [i for i in 1:ndims if i≠cdim && i≠bdim]
    rdims = ntuple(i -> dvec[i], ndims-2)
    return shape, rdims
end


"""
# Explanation
Applies mean and variance normalization over a N-dimensional input `x`. Suppose `x` has
shape (C, `W1,W2,...,Wd`, B), and reshape `x` to shape (C, `T`, B), Each element of `μ` and `σ`
are collected from each channle of all samples, i.e.
+ μ[c,b] = mean(x[c, 1:T, b]), c ∈ 1:C and b ∈ 1:B
+ σ[c,b] =  std(x[c, 1:T, b]), c ∈ 1:C and b ∈ 1:B
finally, the shape of statistical `μ` or `σ` is (C, `1,1,...,1`, B).
## A Picture Sample Has RGB Channles
    ┌┬┬┬┬┬┬┬┐R Channel
    ├┼┼┼┼┼┼┼┤
    H┼┼┼┼┼┼┼┤    ────────────────────────┐
    ├┼┼┼┼┼┼┼┤                            │
    └┴┴┴W┴┴┴┘                            │
    ┌┬┬┬┬┬┬┬┐G Channel                   ▼
    ├┼┼┼┼┼┼┼┤                            ▓
    H┼┼┼┼┼┼┼┤    ──────────────────────► █ μ or σ (shape of 1×3)
    ├┼┼┼┼┼┼┼┤                            ▓
    └┴┴┴W┴┴┴┘                            ▲
    ┌┬┬┬┬┬┬┬┐B Channel                   │
    ├┼┼┼┼┼┼┼┤                            │
    H┼┼┼┼┼┼┼┤    ────────────────────────┘
    ├┼┼┼┼┼┼┼┤
    └┴┴┴W┴┴┴┘
# Constructor
    InstanceNorm(;ndims    :: Int,     # data dimentions
                  cdim     :: Int,     # channle dim
                  bdim     :: Int,     # batch dim
                  channels :: Int,     # channle size
                  reuse    :: Bool,    # if track the running var and mean, then true
                  inertia  :: AbstractFloat=0.9,    # historical proportion
                  eps      :: AbstractFloat=1e-38,  # avoid 0-dividing
                  type     :: Type=Array{Float32})

# N-dimentional Constructor
+ InstanceNorm0d(channels::Int; reuse=false, inertia=0.9, eps=1e-38, type=Array{Float32}) where D
+ InstanceNorm1d(channels::Int; reuse=false, inertia=0.9, eps=1e-38, type=Array{Float32}) where D
+ InstanceNorm2d(channels::Int; reuse=false, inertia=0.9, eps=1e-38, type=Array{Float32}) where D
+ InstanceNorm3d(channels::Int; reuse=false, inertia=0.9, eps=1e-38, type=Array{Float32}) where D
+ InstanceNorm4d(channels::Int; reuse=false, inertia=0.9, eps=1e-38, type=Array{Float32}) where D
+ InstanceNorm5d(channels::Int; reuse=false, inertia=0.9, eps=1e-38, type=Array{Float32}) where D
"""
mutable struct InstanceNorm <: Normalizer
    γ     :: VarOrNil                    # scaling params
    β     :: VarOrNil                    # shifting params
    μ     :: Union{AbstractArray,Nil}    # running average
    σ²    :: Union{AbstractArray,Nil}    # running variance
    ϵ     :: AbstractFloat               # prevent dividing by zero, 1e-38 for default
    ρ     :: AbstractFloat               # inertia coefficient
    dims  :: NTuple                      # dims to collect elements for mean and var
    bdim  :: Int                         # batch-dim
    reuse :: Bool                        # if keep tracking running average and variance
    function InstanceNorm(;ndims    :: Int,     # how many dimentions the input data has
                           cdim     :: Int,     # channle dim
                           bdim     :: Int,     # batch dim
                           channels :: Int,     # channle size
                           reuse    :: Bool,
                           inertia  :: AbstractFloat=0.9,    # smoothing const or historical inertia
                           eps      :: AbstractFloat=1e-38,
                           type     :: Type=Array{Float32})

        shape, dims2reduce = in_shape_and_rdims(ndims, cdim, bdim, channels)
        γ  = Variable{type}( Ones(type, shape), true, true, true)
        β  = Variable{type}(Zeros(type, shape), true, true, true)
        μ  = reuse ? Zeros(type, shape) : nothing
        σ² = reuse ?  Ones(type, shape) : nothing
        ρ  = eltype(type)(inertia)
        new(γ, β, μ, σ², eps, ρ, dims2reduce, bdim,reuse)
    end
    function InstanceNorm(ϵ::Real, ρ::Real, dims::Union{Tuple,Int}, bdim::Int, reuse::Bool)
        new(nothing, nothing, nothing, nothing, ϵ, ρ, dims, bdim, reuse)
    end
end



function forward(I::InstanceNorm, x::Variable)
    γ = I.γ
    β = I.β

    x̌, μ, σ² = znorm_mean_var(x, dims=I.dims, eps=I.ϵ)
    y = x̌ .* γ .+ β
    if I.reuse
        T = eltype(x)
        l = T(1)
        ρ = T(I.ρ)
        𝝁  = I.μ
        𝝈² = I.σ²
        @sync begin
            @async 𝝁  .= ρ .* 𝝁  .+ (l .- ρ) .* mean(μ,  dims=I.bdim)  # running mean
            @async 𝝈² .= ρ .* 𝝈² .+ (l .- ρ) .* mean(σ², dims=I.bdim)  # running var
        end
    end
    return y
end


function predict(I::InstanceNorm, x::AbstractArray)
    γ  = ᵛ(I.γ)
    β  = ᵛ(I.β)

    if !I.reuse
        x̌ = znorm(x, dims=I.dims, eps=I.ϵ)
        y = x̌ .* γ .+ β
    else
        l  = eltype(x)(1)
        μ  = I.μ
        σ² = I.σ²
        x̌  = @. (x - μ) * (l / sqrt(σ²))
        y  = @. x̌ * γ + β
    end
    return y
end


function clone(this::InstanceNorm; type::Type=Array{Float32})
    cloned    = ZNorm(this.ϵ, this.ρ, this.dims, this.bdim, this.reuse)
    cloned.γ  = clone(this.γ, type=type)
    cloned.β  = clone(this.β, type=type)
    cloned.μ  = type(this.μ)
    cloned.σ² = type(this.σ²)
    return cloned
end

function Base.show(io::IO, in::InstanceNorm)
    S = length(in.β.value)
    T = typeof(in.β.value)
    D =  ndims(in.β.value) - 2
    ρ = in.ρ
    print(io, "InstanceNorm$(D)d(channels=$S, inertia=$ρ; reuse=$(in.reuse), type=$T)")
end

function paramsof(bn::InstanceNorm)
    params = Vector{Variable}(undef,2)
    params[1] = bn.γ
    params[2] = bn.β
    return params
end

function xparamsof(in::InstanceNorm)
    xparams = Vector{XVariable}(undef,2)
    xparams[1] = ('w', in.γ)
    xparams[2] = ('b', in.β)
    return xparams
end

function nparamsof(in::InstanceNorm)
    if in.reuse
        return 4*length(in.β)
    else
        return 2*length(in.β)
    end
end

elsizeof(in::InstanceNorm) = elsizeof(in.γ)

function bytesof(in::InstanceNorm, unit::String="MB")
    n = nparamsof(in) * elsizeof(in)
    return blocksize(n, uppercase(unit))
end


for DIMS in [0,1,2,3,4,5]
    @eval begin
        export $(Symbol("InstanceNorm$(DIMS)d"))
        function $(Symbol("InstanceNorm$(DIMS)d"))(channels :: Int;
                                                   reuse    :: Bool=false,
                                                   eps      :: AbstractFloat=1e-38,
                                                   inertia  :: AbstractFloat=0.9,
                                                   type     :: Type=Array{Float32})
            return InstanceNorm(;ndims = $(DIMS+2),
                                 bdim  = $(DIMS+2),
                                 cdim  = 1,
                                 channels, reuse, eps, inertia, type)
        end
    end
end
