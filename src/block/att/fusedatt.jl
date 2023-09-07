export FusedAttention

mutable struct FusedAttention <: Block
    Wq   :: Conv1x1
    Wk   :: Conv1x1
    Wv   :: Conv1x1
    kdim :: Int
    vdim :: Int
    function FusedAttention(ichs::Int; kdim::Int, vdim::Int)
        wq = Conv1x1(ichs, kdim, nothing)
        wk = Conv1x1(ichs, kdim, nothing)
        wv = Conv1x1(ichs, vdim, nothing)
        new(wq, wk, wv, kdim, vdim)
    end
end

function Base.show(io::IO, ::MIME"text/plain", f::FusedAttention)
    println("FusedAttention(kdim=$(f.kdim), vdim=$(f.vdim))")
    println("   WQ is ", f.Wq)
    println("   WK is ", f.Wk)
    println("   WV is ", f.Wv)
end


function forward(att::FusedAttention, xs::Vector{Variable{T}}) where T
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
    D = ndims(first(xs))

    sizex = size(first(xs))     # sizex
    width = prod(sizex[2:D-1])  # ∏ⱼWⱼ
    batch = sizex[D]            # batchsize
    dᴷ = att.kdim               # query dims
    dⱽ = att.vdim               # value dims

    # Suppose each input has size like (C, W₁,W₂,...,Wₘ, B), we will later
    # reshape each of them into (C*∏ⱼWⱼ,1,B)，just like embedding slice in NLP
    kshape = ntuple(3) do i
        isequal(i, 1) && return width * dᴷ
        isequal(i, 2) && return 1
        return batch
    end
    vshape = ntuple(3) do i
        isequal(i, 1) && return width * dⱽ
        isequal(i, 2) && return 1
        return batch
    end

    Qs = Vector{Variable{T}}(undef, L)
    Ks = Vector{Variable{T}}(undef, L)
    Vs = Vector{Variable{T}}(undef, L)
    Threads.@threads for i in 1:L
        Qs[i] = reshape(att.Wq(xs[i]), kshape)
        Ks[i] = reshape(att.Wk(xs[i]), kshape)
        Vs[i] = reshape(att.Wv(xs[i]), vshape)
    end

    # concate L inputs, each with size (C*∏ⱼWⱼ,1,B), into a
    # whole tensor whose size is (C*∏ⱼWⱼ,L,B), then split it
    # into #B tensors, each with size (C*∏ⱼWⱼ, L)
    Q = chunk(cat(Qs, dims=2), dim=3)
    K = chunk(cat(Ks, dims=2), dim=3)
    V = chunk(cat(Vs, dims=2), dim=3)

    αs = Vector{Variable{T}}(undef, batch)  # attention matrices
    ys = Vector{Variable{T}}(undef, batch)  # fused samples
    Threads.@threads for b in 1:batch
        score = softmax(K[b]' * Q[b] / sqrt(dᴷ), dims=1)
        αs[b] = mean(score, dims=2)
        ys[b] = reshape(V[b] * αs[b], width*dⱽ, 1, 1)
    end

    shape = ntuple(N) do i
        isequal(i, 1) && return dⱽ
        isequal(i, D) && return batch
        return sizex[i]
    end
    return reshape(cat(ys, dims=3), shape), αs
end



function predict(att::FusedAttention, xs::Vector{AbstractArry})
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
    D = ndims(first(xs))

    sizex = size(first(xs))     # sizex
    width = prod(sizex[2:D-1])  # ∏ⱼWⱼ
    batch = sizex[D]            # batchsize
    dᴷ = att.kdim               # query dims
    dⱽ = att.vdim               # value dims

    # Suppose each input has size like (C, W₁,W₂,...,Wₘ, B), we will later
    # reshape each of them into (C*∏ⱼWⱼ,1,B)，just like embedding slice in NLP
    kshape = ntuple(3) do i
        isequal(i, 1) && return width * dᴷ
        isequal(i, 2) && return 1
        return batch
    end
    vshape = ntuple(3) do i
        isequal(i, 1) && return width * dⱽ
        isequal(i, 2) && return 1
        return batch
    end

    Qs = Vector{AbstractArry}(undef, L)
    Ks = Vector{AbstractArry}(undef, L)
    Vs = Vector{AbstractArry}(undef, L)
    Threads.@threads for i in 1:L
        Qs[i] = reshape(att.Wq(xs[i]), kshape)
        Ks[i] = reshape(att.Wk(xs[i]), kshape)
        Vs[i] = reshape(att.Wv(xs[i]), vshape)
    end

    # concate L inputs, each with size (C*∏ⱼWⱼ,1,B), into a
    # whole tensor whose size is (C*∏ⱼWⱼ,L,B), then split it
    # into #B tensors, each with size (C*∏ⱼWⱼ, L)
    Q = chunk(cat(Qs, dims=2), dim=3)
    K = chunk(cat(Ks, dims=2), dim=3)
    V = chunk(cat(Vs, dims=2), dim=3)

    αs = Vector{AbstractArry}(undef, batch)  # attention matrices
    ys = Vector{AbstractArry}(undef, batch)  # fused samples
    Threads.@threads for b in 1:batch
        score = softmax(K[b]' * Q[b] / sqrt(dᴷ), dims=1)
        αs[b] = mean(score, dims=2)
        ys[b] = reshape(V[b] * αs[b], width*dⱽ, 1, 1)
    end

    shape = ntuple(N) do i
        isequal(i, 1) && return dⱽ
        isequal(i, D) && return batch
        return sizex[i]
    end
    return reshape(cat(ys, dims=3), shape), αs
end
