mutable struct SparseMasking <: Block
    ratio::AbstractFloat
    scale::VarOrNil
    views::Tuple
    function SparseMasking(ratio::AbstractFloat;
                           ndims::Int,
                           keptdim::Int,
                           keptsize::Int,
                           type::Type=Array{Float32})
        @assert ndims > 0 "ndims > 0, but got ndims=$ndims"
        @assert ndims >= keptdim >= 1 "1 ≤ keptdim ≤ $ndims, but got keptdim=$keptdim"
        dtype = eltype(type)
        views = ntuple(i -> i, ndims);
        shape = ntuple(i -> i==keptdim ? keptsize : 1, ndims);
        scale = Variable{type}(randn(dtype, shape), true, true, true);
        new(dtype(ratio), scale, views)
    end
    function SparseMasking(ratio::AbstractFloat, views::Tuple)
        new(ratio, nothing, views)
    end
end


function sparseLoss(s::SparseMasking)
    @assert 0. < s.ratio < 1.
    TO = eltype(s.ratio)
    ϵ  = TO(1e-38)
    𝟙  = TO(1.0f0)
    N  = 1 / length(s.scale)

    𝜌  = s.ratio                                   # sparsity ratio
    c  = 𝜌 * log(𝜌) + (1 - 𝜌) * log(1 - 𝜌)         # - info entropy
    p  = sum(sigmoid(s.scale), dims=s.views) * N   #   active ratio
    t₁ = -      𝜌  .* log.(     ᵛ(p) .+ ϵ) # cross entropy part one
    t₂ = - (𝟙 - 𝜌) .* log.(𝟙 .- ᵛ(p) .+ ϵ) # cross entropy part two
    y  = typeof(s.scale)(t₁ + t₂ .+ c, p.backprop)

    if y.backprop
        y.backward = function sparseLossBackward()
            if need2computeδ!(p)
                δ₁ =  (𝟙 - 𝜌) ./ (𝟙 .- ᵛ(p) .+ ϵ)
                δ₂ =       𝜌  ./ (     ᵛ(p) .+ ϵ)
                δ(p) .+= δ(y) .* (δ₁ - δ₂)
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, p)
    end
    return y
end


function clone(this::SparseMasking; type::Type=Array{Float32})
    cloned = SparseMasking(this.ratio, this.views)
    cloned.scale = clone(this.scale, type=type)
    return cloned
end


function Base.show(io::IO, s::SparseMasking)
    print(io, "SparseMasking(ratio=$(s.ratio))")
end


function paramsof(s::SparseMasking)
    params = Vector{Variable}(undef,1)
    params[1] = s.scale
    return params
end


function xparamsof(s::SparseMasking)
    xparams = Vector{XVariable}(undef,1)
    xparams[1] = ('o', s.scale) # other labeled
    return xparams
end


function nparamsof(s::SparseMasking)
    return length(s.scale)
end


function bytesof(s::SparseMasking, unit::String="MB")
    return blocksize(sizeof(s.scale), uppercase(unit))
end


function forward(s::SparseMasking, x::Variable{T}) where T
    mask = sigmoid(s.scale)
    return mask .* x
end


function predict(s::SparseMasking, x::AbstractArray)
    mask = sigmoid(s.scale.value)
    return mask .* x
end
