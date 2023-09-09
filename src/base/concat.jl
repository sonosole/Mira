"""
    cat(x₁::Variable{T}, x₂::Variable{T}; dims::Int=1) where T
Concatenate along dimension `dims`.
# Example
```
julia> x1 = Variable(1ones(2,1));
julia> x2 = Variable(2ones(2,2));
julia> x12 = cat(x1,x2; dims=2)
 None Leaf's value is 2×3 Matrix{Float32}:
 1.0  2.0  2.0
 1.0  2.0  2.0
```
"""
function Base.cat(x₁::Variable{T}, x₂::Variable{T}; dims::Int=1) where T
    D      = ndims(x₁)
    sizex₁ = size(x₁)
    sizex₂ = size(x₂)

    for d in 1:D
        isequal(d, dims) && continue
        if !isequal(sizex₁[d], sizex₂[d])
            error("Dimention mismatch at $d-dim")
        end
    end

    c₁ = ntuple(i -> 1:sizex₁[i], D)
    c₂ = ntuple(D) do i
        i ≠ dims && return 1:sizex₂[i]
        offset = sizex₁[i]
        return (1 + offset):(sizex₂[i] + offset)
    end
    sizex = ntuple(D) do i
        i ≠ dims && return sizex₁[i]
        return sizex₁[i] + sizex₂[i]
    end

    v = similar(ᵛ(x₁), sizex)
    @sync begin
        Threads.@spawn v[c₁...] .= ᵛ(x₁)
        Threads.@spawn v[c₂...] .= ᵛ(x₂)
    end
    x = Variable{T}(v, x₁.backprop || x₂.backprop)

    if x.backprop
        x.backward = function ∇vcat()
            @sync begin
                Threads.@spawn (needgrad(x₁) && (x₁ ← δ(x)[c₁...]))
                Threads.@spawn (needgrad(x₂) && (x₂ ← δ(x)[c₂...]))
            end
            ifNotKeepδThenFreeδ!(x)
        end
        addchild(x, x₁)
        addchild(x, x₂)
    end
    return x
end


"""
    cat(x₁::Variable{T}, x₂::Variable{T}, x₃::Variable{T}; dims::Int=1) where T
Concatenate along dimension `dims`.
# Example
```
julia> x1 = Variable(1ones(2,1));
julia> x2 = Variable(2ones(2,2));
julia> x3 = Variable(3ones(2,3));
julia> x123 = cat(x1,x2,x3;dims=2)
 None Leaf's value is 2×6 Matrix{Float32}:
 1.0  2.0  2.0  3.0  3.0  3.0
 1.0  2.0  2.0  3.0  3.0  3.0
```
"""
function Base.cat(x₁::Variable{T}, x₂::Variable{T}, x₃::Variable{T}; dims::Int=1) where T
    D      = ndims(x₁)
    sizex₁ = size(x₁)
    sizex₂ = size(x₂)
    sizex₃ = size(x₃)

    for d in 1:D
        isequal(d, dims) && continue
        if sizex₁[d] ≠ sizex₂[d] ≠ sizex₃[d]
            error("Dimention mismatch at $d-dim")
        end
    end

    c₁ = ntuple(i -> 1:sizex₁[i], D)
    c₂ = ntuple(D) do i
        i ≠ dims && return 1:sizex₂[i]
        offset = sizex₁[i]
        return (1 + offset):(sizex₂[i] + offset)
    end
    c₃ = ntuple(D) do i
        i ≠ dims && return 1:sizex₃[i]
        offset = sizex₁[i] + sizex₂[i]
        return (1 + offset):(sizex₃[i] + offset)
    end

    sizex = ntuple(D) do i
        i ≠ dims && return sizex₁[i]
        return sizex₁[i] + sizex₂[i] + sizex₃[i]
    end

    v = similar(ᵛ(x₁), sizex)
    @sync begin
        Threads.@spawn v[c₁...] .= ᵛ(x₁)
        Threads.@spawn v[c₂...] .= ᵛ(x₂)
        Threads.@spawn v[c₃...] .= ᵛ(x₃)
    end
    x = Variable{T}(v, x₁.backprop || x₂.backprop || x₃.backprop)

    if x.backprop
        x.backward = function ∇vcat()
            @sync begin
                Threads.@spawn (needgrad(x₁) && (x₁ ← δ(x)[c₁...]))
                Threads.@spawn (needgrad(x₂) && (x₂ ← δ(x)[c₂...]))
                Threads.@spawn (needgrad(x₃) && (x₃ ← δ(x)[c₃...]))
            end
            ifNotKeepδThenFreeδ!(x)
        end
        addchild(x, x₁)
        addchild(x, x₂)
        addchild(x, x₃)
    end
    return x
end


"""
    cat(xs::Vector{Variable{T}}; dims::Int=1) where T
Concatenate along dimension `dims`.
# Example
```
julia> x1 = Variable(1ones(2,1));
julia> x2 = Variable(2ones(2,2));
julia> x3 = Variable(3ones(2,3));
julia> x4 = Variable(4ones(2,4));
julia> x1234 = cat([x1,x2,x3,x4];dims=2)
 None Leaf's value is 2×10 Matrix{Float32}:
 1.0  2.0  2.0  3.0  3.0  3.0  4.0  4.0  4.0  4.0
 1.0  2.0  2.0  3.0  3.0  3.0  4.0  4.0  4.0  4.0
```
"""
function Base.cat(xs::Vector{Variable{T}}; dims::Int=1) where T
    N = length(xs)
    isequal(N, 1) && return first(xs)
    D = ndims(first(xs))

    sizexs = Vector{Dims{D}}(undef, N)
    sizexs[1] = size(first(xs))
    for n in 2:N
        sizexs[n] = size(xs[n])
        for d in 1:D
            isequal(d, dims) && continue
            if sizexs[n][d] ≠ sizexs[n-1][d]
                error("Dimention mismatch at $d-dim")
            end
        end
    end

    cs = Vector{CartesianIndices}(undef, N)
    for n in 1:N
        cⁿ = ntuple(D) do i
            i ≠ dims && return 1:sizexs[n][i]
            offset = 0
            for k in 1:n-1
                offset += sizexs[k][i]
            end
            return (1 + offset):(sizexs[n][i] + offset)
        end
        cs[n] = CartesianIndices(cⁿ)
    end

    sizex = ntuple(D) do i
        i ≠ dims && return sizexs[1][i]
        d = 0
        for n in 1:N
            d += sizexs[n][i]
        end
        return d
    end

    v = similar(ᵛ(first(xs)), sizex)
    @sync begin
        Threads.@threads for n in 1:N
            v[cs[n]] .= ᵛ(xs[n])
        end
    end
    x = Variable{T}(v, any(xs[n].backprop for n in 1:N))

    if x.backprop
        x.backward = function ∇vcat()
            @sync begin
                Threads.@threads for n in 1:N
                    (needgrad(xs[n]) && (xs[n] ← δ(x)[cs[n]]))
                end
            end
            ifNotKeepδThenFreeδ!(x)
        end
        for n in 1:N
            addchild(x, xs[n])
        end
    end
    return x
end


"""
    vcat(x₁::Variable{T}, x₂::Variable{T}) where T
Concatenate along dimension 1.
# Example
```julia
julia> x1 = Variable(1ones(1,4));
julia> x2 = Variable(2ones(2,4));
julia> x12 = vcat(x1,x2)
 None Leaf's value is 3×4 Matrix{Float32}:
 1.0  1.0  1.0  1.0
 2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0
```
"""
function Base.vcat(x₁::Variable{T}, x₂::Variable{T}) where T
    return cat(x₁, x₂; dims=1)
end


"""
    vcat(x₁::Variable{T}, x₂::Variable{T}, x₃::Variable{T}) where T
Concatenate along dimension 1.
# Example
```
julia> x1 = Variable(1ones(1,4));
julia> x2 = Variable(2ones(2,4));
julia> x3 = Variable(3ones(3,4));
julia> x123 = vcat(x1,x2,x3)
 None Leaf's value is 6×4 Matrix{Float32}:
 1.0  1.0  1.0  1.0
 2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0
 3.0  3.0  3.0  3.0
 3.0  3.0  3.0  3.0
 3.0  3.0  3.0  3.0
```
"""
function Base.vcat(x₁::Variable{T}, x₂::Variable{T}, x₃::Variable{T}) where T
    return cat(x₁, x₂, x₃; dims=1)
end


"""
    vcat(x₁::Variable{T}, x₂::Variable{T}, x₃::Variable{T}) where T
Concatenate along dimension 1.
# Example
```
julia> x1 = Variable(1ones(1,4));
julia> x2 = Variable(2ones(2,4));
julia> x3 = Variable(3ones(3,4));
julia> x4 = Variable(4ones(4,4));
julia> x1234 = vcat([x1,x2,x3,x4])
 None Leaf's value is 10×4 Matrix{Float32}:
 1.0  1.0  1.0  1.0
 2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0
 3.0  3.0  3.0  3.0
 3.0  3.0  3.0  3.0
 3.0  3.0  3.0  3.0
 4.0  4.0  4.0  4.0
 4.0  4.0  4.0  4.0
 4.0  4.0  4.0  4.0
 4.0  4.0  4.0  4.0
```
"""
function Base.vcat(xs::Vector{Variable{T}}) where T
    return cat(xs; dims=1)
end


"""
    hcat(x₁::Variable{T}, x₂::Variable{T}) where T
Concatenate along dimension 2.
"""
function Base.hcat(x₁::Variable{T}, x₂::Variable{T}) where T
    return cat(x₁, x₂; dims=2)
end

"""
    hcat(x₁::Variable{T}, x₂::Variable{T}, x₃::Variable{T})
Concatenate along dimension 2.
"""
function Base.hcat(x₁::Variable{T}, x₂::Variable{T}, x₃::Variable{T}) where T
    return cat(x₁, x₂, x₃; dims=2)
end

"""
    hcat(xs::Vector{Variable{T}}) where T
Concatenate along dimension 2.
"""
function Base.hcat(xs::Vector{Variable{T}}) where T
    return cat(xs; dims=2)
end



function Base.cat(xs::Vector{T}; dims::Int=1) where T <: AbstractArray
    N = length(xs)
    isequal(N, 1) && return first(xs)
    D = ndims(first(xs))

    sizexs = Vector{Dims{D}}(undef, N)
    sizexs[1] = size(first(xs))
    for n in 2:N
        sizexs[n] = size(xs[n])
        for d in 1:D
            isequal(d, dims) && continue
            if sizexs[n][d] ≠ sizexs[n-1][d]
                error("Dimention mismatch at $d-dim")
            end
        end
    end

    cs = Vector{CartesianIndices}(undef, N)
    for n in 1:N
        cⁿ = ntuple(D) do i
            i ≠ dims && return 1:sizexs[n][i]
            offset = 0
            for k in 1:n-1
                offset += sizexs[k][i]
            end
            return (1 + offset):(sizexs[n][i] + offset)
        end
        cs[n] = CartesianIndices(cⁿ)
    end

    sizex = ntuple(D) do i
        i ≠ dims && return sizexs[1][i]
        d = 0
        for n in 1:N
            d += sizexs[n][i]
        end
        return d
    end

    x = similar(first(xs), sizex)
    @sync begin
        Threads.@threads for n in 1:N
            x[cs[n]] .= xs[n]
        end
    end
    return x
end

function Base.vcat(xs::Vector{T}) where T <: AbstractArray
    return cat(xs; dims=1)
end

function Base.hcat(xs::Vector{T}) where T <: AbstractArray
    return cat(xs; dims=2)
end
