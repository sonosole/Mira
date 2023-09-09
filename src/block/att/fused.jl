export FusedAttention
const QKVChanger = Union{Function, Nothing, Block}

mutable struct FusedAttention <: Block
    Qf :: ComposedFunction
    Kf :: ComposedFunction
    Vf :: ComposedFunction
    function FusedAttention(kinner::QKVChanger,
                            kouter::QKVChanger,
                            vinner::QKVChanger,
                            vouter::QKVChanger)
        Fq = kouter ∘ kinner
        Fk = kouter ∘ kinner
        Fv = vouter ∘ vinner
        new(Fq, Fk, Fv)
    end
end

function Base.show(io::IO, ::MIME"text/plain", f::FusedAttention)
    println("FusedAttention: ")
    println("   Query Operator: ", f.Qf.outer, " ∘ ",f.Qf.inner)
    println("   Key   Operator: ", f.Kf.outer, " ∘ ",f.Kf.inner)
    println("   Value Operator: ", f.Vf.outer, " ∘ ",f.Vf.inner)
end


function forward(att::FusedAttention, xs::Vector{Variable{T}}) where T
    # first we produce xs using any ComposedFunction e.g.
    # 1. Conv1x1 ∘ Pool3d ∘ softmax, for 3-Dim tensor
    # 2. Conv2d ∘ nothing, with strides > 1 for dowsampling, and smaller output channel dimensions
    # 3. nothing ∘ nothing, just keeping the original data
    L  = length(xs)
    Qs = Vector{Variable{T}}(undef, L)
    Ks = Vector{Variable{T}}(undef, L)
    Vs = Vector{Variable{T}}(undef, L)
    Threads.@threads for i in 1:L
        Qs[i] = att.Qf(xs[i])
        Ks[i] = att.Kf(xs[i])
        Vs[i] = att.Vf(xs[i])
    end

    S = size(first(Vs))        # backup for recovering shape
    N = ndims(first(Vs))       # ─┬─►  N may equal to D, if elements in Qs Ks
    may_reshape_qkv_to_3d(Qs)  #  │    and Vs have more then two dimensions.
    may_reshape_qkv_to_3d(Ks)  #  │    in this case, may_reshape_qkv_to_3d fn
    may_reshape_qkv_to_3d(Vs)  #  │    would do nothing to elements in Qs, Ks
    D = ndims(first(Vs))       # ─┘    and Vs.

    sizek = size(first(Ks))
    sizev = size(first(Vs))
    widthk = prod(sizek[2:D-1])  # ∏ⱼWᴷⱼ, j in 2:D-1
    widthv = prod(sizev[2:D-1])  # ∏ⱼWⱽⱼ, j in 2:D-1
    batch  = sizev[D]     # batchsize
    dᴷ = first(sizek)     # query dims
    dⱽ = first(sizev)     # value dims

    # Suppose each input has size like (C, W₁,W₂,...,Wₘ, B), we will later
    # reshape each of them into (C*∏ⱼWⱼ,1,B)，just like embedding slice in NLP
    kshape = ntuple(3) do i
        isequal(i, 1) && return widthk * dᴷ
        isequal(i, 2) && return 1
        return batch
    end
    vshape = ntuple(3) do i
        isequal(i, 1) && return widthv * dⱽ
        isequal(i, 2) && return 1
        return batch
    end

    Threads.@threads for i in 1:L
        Qs[i] = reshape(Qs[i], kshape)
        Ks[i] = reshape(Ks[i], kshape)
        Vs[i] = reshape(Vs[i], vshape)
    end

    # concate L inputs of size (C*∏ⱼWⱼ, 1, B) into a
    # whole tensor with size   (C*∏ⱼWⱼ, L, B), then split it
    # into B tensors of size   (C*∏ⱼWⱼ, L)
    Q = chunk(cat(Qs, dims=2), dim=3)
    K = chunk(cat(Ks, dims=2), dim=3)
    V = chunk(cat(Vs, dims=2), dim=3)

    αs = Vector{Variable{T}}(undef, batch)  # attention matrices
    ys = Vector{Variable{T}}(undef, batch)  # fused samples
    Threads.@threads for b in 1:batch
        score = softmax(K[b]' * Q[b] / sqrt(dᴷ), dims=1)
        αs[b] = mean(score, dims=2)
        ys[b] = reshape(V[b] * αs[b], widthv * dⱽ, 1, 1)
    end

    shape = ntuple(N) do i
        isequal(i, 1) && return dⱽ
        isequal(i, D) && return batch
        return S[i]
    end
    return reshape(cat(ys, dims=3), shape)
end


function predict(att::FusedAttention, xs::Vector{T}) where T <: AbstractArray
    # first we produce xs using any ComposedFunction e.g.
    # 1. Conv1x1 ∘ Pool3d ∘ softmax, for 3-Dim tensor
    # 2. Conv2d ∘ nothing, with strides > 1 for dowsampling, and smaller output channel dimensions
    # 3. nothing ∘ nothing, just keeping the original data
    L  = length(xs)
    Qs = Vector{AbstractArray}(undef, L)
    Ks = Vector{AbstractArray}(undef, L)
    Vs = Vector{AbstractArray}(undef, L)
    Threads.@threads for i in 1:L
        Qs[i] = att.Qf(xs[i])
        Ks[i] = att.Kf(xs[i])
        Vs[i] = att.Vf(xs[i])
    end

    S = size(first(Vs))        # backup for recovering shape
    N = ndims(first(Vs))       # ─┬─►  N may equal to D, if elements in Qs Ks
    may_reshape_qkv_to_3d(Qs)  #  │    and Vs have more then two dimensions.
    may_reshape_qkv_to_3d(Ks)  #  │    in this case, may_reshape_qkv_to_3d fn
    may_reshape_qkv_to_3d(Vs)  #  │    would do nothing to elements in Qs, Ks
    D = ndims(first(Vs))       # ─┘    and Vs.

    sizek = size(first(Ks))
    sizev = size(first(Vs))
    widthk = prod(sizek[2:D-1])  # ∏ⱼWᴷⱼ, j in 2:D-1
    widthv = prod(sizev[2:D-1])  # ∏ⱼWⱽⱼ, j in 2:D-1
    batch  = sizev[D]     # batchsize
    dᴷ = first(sizek)     # query dims
    dⱽ = first(sizev)     # value dims

    # Suppose each input has size like (C, W₁,W₂,...,Wₘ, B), we will later
    # reshape each of them into (C*∏ⱼWⱼ,1,B)，just like embedding slice in NLP
    kshape = ntuple(3) do i
        isequal(i, 1) && return widthk * dᴷ
        isequal(i, 2) && return 1
        return batch
    end
    vshape = ntuple(3) do i
        isequal(i, 1) && return widthv * dⱽ
        isequal(i, 2) && return 1
        return batch
    end

    Threads.@threads for i in 1:L
        Qs[i] = reshape(Qs[i], kshape)
        Ks[i] = reshape(Ks[i], kshape)
        Vs[i] = reshape(Vs[i], vshape)
    end

    # concate L inputs of size (C*∏ⱼWⱼ, 1, B) into a
    # whole tensor with size   (C*∏ⱼWⱼ, L, B), then split it
    # into B tensors of size   (C*∏ⱼWⱼ, L)
    Q = chunk(cat(Qs, dims=2), dim=3)
    K = chunk(cat(Ks, dims=2), dim=3)
    V = chunk(cat(Vs, dims=2), dim=3)

    αs = Vector{AbstractArray}(undef, batch)  # attention matrices
    ys = Vector{AbstractArray}(undef, batch)  # fused samples
    d⁻¹= eltype(T)(1 / sqrt(dᴷ))
    Threads.@threads for b in 1:batch
        score = softmax(K[b]' * Q[b] .* d⁻¹, dims=1)
        αs[b] = mean(score, dims=2)
        ys[b] = reshape(V[b] * αs[b], widthv * dⱽ, 1, 1)
    end

    shape = ntuple(N) do i
        isequal(i, 1) && return dⱽ
        isequal(i, D) && return batch
        return S[i]
    end
    return reshape(cat(ys, dims=3), shape)
end


function may_reshape_qkv_to_3d(xs::Union{Vector{Variable{T}}, Vector{S}}) where {T,S <: AbstractArray}
    L = length(xs)
    x = first(xs)
    N = ndims(x)
    if isequal(N, 1)
        # ensure ndims(x) ≥ 3,
        # if not, then reshape.
        C = size(x, 1)
        Threads.@threads for i in 1:L
            xs[i] = reshape(xs[i], C, 1, 1)
        end
    elseif isequal(N, 2)
        C = size(x, 1)
        B = size(x, 2)
        Threads.@threads for i in 1:L
            xs[i] = reshape(xs[i], C, 1, B)
        end
    end
    return nothing
end
