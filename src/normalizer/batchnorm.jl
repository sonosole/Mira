export BatchNorm

"""
# Constructor
    BatchNorm(;ndims::Int,                     # dimentions the input data has
               keptdims::Union{Tuple,Int},     # must be unique and sorted and positive
               keptsize::Union{Tuple,Int},     # must be positive
               momentum::AbstractFloat=0.9,    # smoothing const or historical inertia
               eps::AbstractFloat=1e-38,
               type::Type=Array{Float32})

Applies mean and variance normalization over a N-dimensional input
"""
mutable struct BatchNorm <: Normalizer
    γ  :: VarOrNil                        # scaling params
    β  :: VarOrNil                        # shifting params
    μ  :: Union{AbstractArray,Nil}        # running average
    σ² :: Union{AbstractArray,Nil}        # running variance otherwise standard deviation
    views   :: NTuple                     # views to collect elements for mean and var
    eps     :: AbstractFloat              # prevent dividing by zero, 1e-38 for default
    inertia :: AbstractFloat              # inertia coefficient
    function BatchNorm(;ndims::Int,       # how many dimentions the input data has
                        keptdims::Union{Tuple,Int},     # must be unique and sorted and positive
                        keptsize::Union{Tuple,Int},     # must be positive
                        momentum::AbstractFloat=0.9,    # smoothing const or historical inertia
                        eps::AbstractFloat=1e-38,
                        type::Type=Array{Float32})

        if length(keptdims) ≠ length(keptsize)
            error("got keptdims=$keptdims, keptsize=$keptsize, they shall be the same length")
        end
        shape, views = ShapeAndViews(ndims, keptdims, keptsize);
        γ  = Variable{type}( Ones(type, shape), true, true, true);
        β  = Variable{type}(Zeros(type, shape), true, true, true);
        σ² =  Ones(type, shape);
        μ  = Zeros(type, shape);
        new(γ, β, μ, σ², views, eps, eltype(type)(momentum))
    end
    function BatchNorm(views, eps, momentum)
        new(nothing, nothing, nothing, nothing, views, eps, momentum)
    end
end



function forward(b::BatchNorm, x::Variable{T}) where T
    γ = b.γ
    β = b.β
    ρ = b.inertia

    x̌, μ, σ² = znorm_mean_var(x, dims=b.views, eps=b.eps)
    y = x̌ .* γ .+ β

    @. b.μ  = ρ * b.μ  + (1 - ρ) * μ    # running mean
    @. b.σ² = ρ * b.σ² + (1 - ρ) * σ²   # running var
    return y
end


function predict(b::BatchNorm, x::AbstractArray)
    γ  = ᵛ(b.γ)
    β  = ᵛ(b.β)
    μ  = b.μ
    σ² = b.σ²

    x̌ = @. (x .- μ) ./ sqrt.(σ²)
    y = @. x̌ .* γ .+ β
    return y
end


function clone(this::BatchNorm; type::Type=Array{Float32})
    cloned    = ZNorm(this.views, this.eps, this.inertia)
    cloned.γ  = clone(this.γ, type=type)
    cloned.β  = clone(this.β, type=type)
    cloned.μ  = type(this.μ)
    cloned.σ² = type(this.σ)
    return cloned
end

function Base.show(io::IO, bn::BatchNorm)
    SIZE = length(bn.β.value)
    TYPE = typeof(bn.β.value)
    dims = length(bn.views) - 1
    rho  = bn.inertia
    print(io, "BatchNorm$(dims)d(channels=$SIZE, momentum=$rho; type=$TYPE)")
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


for dims in [0,1,2,3,4,5]
    @eval begin
        export $(Symbol("BatchNorm$(dims)d"))
        function $(Symbol("BatchNorm$(dims)d"))(nchannels::Int;
                                                eps::AbstractFloat=1e-38,
                                                momentum::AbstractFloat=0.9,
                                                type::Type=Array{Float32})
            return BatchNorm(eps=eps,
                             ndims=$(dims+2),
                             keptdims=1,
                             keptsize=nchannels,
                             momentum=momentum,
                             type=type)
        end
    end
end
