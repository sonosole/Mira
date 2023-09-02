"""
    vcat(x₁::Variable{T}, x₂::Variable{T}) where T
Concatenate along dimension 1.
# Example
```julia
julia> x1 = Variable(1ones(1,4),keepsgrad=true);
julia> x2 = Variable(2ones(2,4),keepsgrad=true);
julia> x12 = vcat(x1,x2)
 None Leaf's value is 3×4 Matrix{Float32}:
 1.0  1.0  1.0  1.0
 2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0
```
"""
function Base.vcat(x₁::Variable{T}, x₂::Variable{T}) where T
    D      = ndims(x₁)
    sizex₁ = size(x₁)
    sizex₂ = size(x₂)

    if !(sizex₁[2:D] == sizex₂[2:D])
        error("Dimention mismatch from 2 to $D dims")
    end

    c₁ = ntuple(i -> 1:sizex₁[i], D)
    c₂ = ntuple(D) do i
        i≠1 && return 1:sizex₂[i]
        offset = sizex₁[1]
        return (1 + offset):(sizex₂[1] + offset)
    end
    sizex = ntuple(D) do i
        i≠1 && return sizex₁[i]
        return sizex₁[1] + sizex₂[1]
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
                Threads.@spawn (need2computeδ!(x₁) && (x₁ ← δ(x)[c₁...]))
                Threads.@spawn (need2computeδ!(x₂) && (x₂ ← δ(x)[c₂...]))
            end
            ifNotKeepδThenFreeδ!(x)
        end
        addchild(x, x₁)
        addchild(x, x₂)
    end
    return x
end


"""
    vcat(x₁::Variable{T}, x₂::Variable{T}, x₃::Variable{T}) where T
Concatenate along dimension 1.
# Example
```
julia> x1 = Variable(1ones(1,4),keepsgrad=true);
julia> x2 = Variable(2ones(2,4),keepsgrad=true);
julia> x3 = Variable(3ones(3,4),keepsgrad=true);
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
    D      = ndims(x₁)
    sizex₁ = size(x₁)
    sizex₂ = size(x₂)
    sizex₃ = size(x₃)

    if !(sizex₁[2:D] == sizex₂[2:D] == sizex₃[2:D])
        error("Dimention mismatch from 2 to $D dims")
    end

    c₁ = ntuple(i -> 1:sizex₁[i], D)
    c₂ = ntuple(D) do i
        i≠1 && return 1:sizex₂[i]
        offset = sizex₁[1]
        return (1 + offset):(sizex₂[1] + offset)
    end
    c₃ = ntuple(D) do i
        i≠1 && return 1:sizex₃[i]
        offset = sizex₁[1] + sizex₂[1]
        return (1 + offset):(sizex₃[1] + offset)
    end

    sizex = ntuple(D) do i
        i≠1 && return sizex₁[i]
        return sizex₁[1] + sizex₂[1] + sizex₃[1]
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
                Threads.@spawn (need2computeδ!(x₁) && (x₁ ← δ(x)[c₁...]))
                Threads.@spawn (need2computeδ!(x₂) && (x₂ ← δ(x)[c₂...]))
                Threads.@spawn (need2computeδ!(x₃) && (x₃ ← δ(x)[c₃...]))
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
    vcat(x₁::Variable{T}, x₂::Variable{T}, x₃::Variable{T}) where T
Concatenate along dimension 1.
# Example
```
julia> x1 = Variable(1ones(1,4),keepsgrad=true);
julia> x2 = Variable(2ones(2,4),keepsgrad=true);
julia> x3 = Variable(3ones(3,4),keepsgrad=true);
julia> x4 = Variable(4ones(4,4),keepsgrad=true);
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
    N = length(xs)
    D = ndims(first(xs))
    @assert N>1 "you don't have to vcat,because you only have one input"
    sizexs = Vector{Dims{D}}(undef, N)
    sizexs[1] = size(first(xs))
    for n in 2:N
        sizexs[n] = size(xs[n])
        if !(sizexs[n][2:D] == sizexs[n-1][2:D])
            error("Dimention mismatch from 2 to $D dims")
        end
    end

    cs = Vector{CartesianIndices}(undef, N)
    for n in 1:N
        cⁿ = ntuple(D) do i
            i≠1 && return 1:sizexs[n][i]
            offset = 0
            for k in 1:n-1
                offset += sizexs[k][1]
            end
            return (1 + offset):(sizexs[n][1] + offset)
        end
        cs[n] = CartesianIndices(cⁿ)
    end

    sizex = ntuple(D) do i
        i≠1 && return sizexs[1][i]
        d = 0
        for n in 1:N
            d += sizexs[n][1]
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
                    (need2computeδ!(xs[n]) && (xs[n] ← δ(x)[cs[n]]))
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
