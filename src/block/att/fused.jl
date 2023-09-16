export FusedAttention
const Fuser = Union{Function, Nothing, Block}

mutable struct FusedAttention <: Block
    Qf :: ComposedFunction
    Kf :: ComposedFunction
    Vf :: ComposedFunction

    poolfn :: Function
    normfn :: Function

    dropdims :: IntOrDims
    droprate :: Real

    need_att_mat :: Bool
    need_att_vec :: Bool
    function FusedAttention(qinner::Fuser, qouter::Fuser,
                            kinner::Fuser, kouter::Fuser,
                            vinner::Fuser, vouter::Fuser;
                            need_att_mat::Bool=false,
                            need_att_vec::Bool=false,
                            poolfn::Function=mean,
                            normfn::Function=softmax,
                            dropdims::IntOrDims=(1,2),
                            droprate::Real=0.0)
        Fq = qouter ∘ qinner
        Fk = kouter ∘ kinner
        Fv = vouter ∘ vinner
        new(Fq, Fk, Fv,
            poolfn,  normfn,
            dropdims, droprate,
            need_att_mat, need_att_vec)
    end
    function FusedAttention(fq::ComposedFunction,
                            fk::ComposedFunction,
                            fv::ComposedFunction,
                            poolf::Function,
                            normf::Function,
                            dropdims::IntOrDims,
                            droprate::Real,
                            need_att_mat::Bool,
                            need_att_vec::Bool)
        new(fq, fk, fv,
            poolf, normf,
            dropdims, droprate,
            need_att_mat, need_att_vec)
    end
end

function clone(this::FusedAttention; type=Array{Float32})
    Qf = clone(this.Qf; type)
    Kf = clone(this.Kf; type)
    Vf = clone(this.Vf; type)
    cloned = FusedAttention(Qf, Kf, Vf,
                            this.poolfn, this.normfn,
                            this.dropdims, this.droprate,
                            this.need_att_mat, this.need_att_vec)
end

function Base.show(io::IO, ::MIME"text/plain", f::FusedAttention)
    println("FusedAttention: ")
    println(" Qf: ", f.Qf.outer, " ∘ ",f.Qf.inner)
    println(" Kf: ", f.Kf.outer, " ∘ ",f.Kf.inner)
    println(" Vf: ", f.Vf.outer, " ∘ ",f.Vf.inner)
    println(" normfn: ", f.normfn, " at dim 1")
    println(" poolfn: ", f.poolfn, " at dim 2")
    println(" need att vec: ", colorbool(f.need_att_vec))
    println(" need att mat: ", colorbool(f.need_att_mat))
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

    normfn = att.normfn
    poolfn = att.poolfn
    mat = Vector{Variable{T}}(undef, batch)  # attention matrices
    vec = Vector{Variable{T}}(undef, batch)  # attention vectors
    ys  = Vector{Variable{T}}(undef, batch)  # fused samples
    Threads.@threads for b in 1:batch
        mat[b] = normfn(K[b]' * Q[b] / sqrt(dᴷ), dims=1)
        chosen = xdropout(mat[b], p=att.droprate, dims=att.dropdims)
        vec[b] = poolfn(chosen, dims=2)
        ys[b]  = reshape(V[b] * vec[b], widthv * dⱽ, 1, 1)
    end

    shape = ntuple(N) do i
        isequal(i, 1) && return dⱽ
        isequal(i, D) && return batch
        return S[i]
    end
    y = reshape(cat(ys, dims=3), shape)

    needvec = att.need_att_vec
    needmat = att.need_att_mat
    if !needvec && !needmat
        return y
    elseif needvec && !needmat
        return y, vec
    elseif !needvec && needmat
        return y, mat
    else
        return y, vec, mat
    end
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

    normfn = att.normfn
    poolfn = att.poolfn
    mat = Vector{AbstractArray}(undef, batch)  # attention matrices
    vec = Vector{AbstractArray}(undef, batch)  # attention vectors
    ys  = Vector{AbstractArray}(undef, batch)  # fused samples
    d⁻¹ = eltype(T)(1 / sqrt(dᴷ))
    Threads.@threads for b in 1:batch
        mat[b] = normfn(K[b]' * Q[b] .* d⁻¹, dims=1)
        vec[b] = poolfn(mat[b], dims=2)
        ys[b]  = reshape(V[b] * vec[b], widthv * dⱽ, 1, 1)
    end

    shape = ntuple(N) do i
        isequal(i, 1) && return dⱽ
        isequal(i, D) && return batch
        return S[i]
    end
    y = reshape(cat(ys, dims=3), shape)

    needvec = att.need_att_vec
    needmat = att.need_att_mat
    if !needvec && !needmat
        return y
    elseif needvec && !needmat
        return y, vec
    elseif !needvec && needmat
        return y, mat
    else
        return y, vec, mat
    end
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


# function may_reshape_qkv_to_3d(x::Union{Variable{T}, S}) where {T,S <: AbstractArray}
#     N = ndims(x)
#     if isequal(N, 1)
#         # ensure ndims(x) ≥ 3,
#         # if not, then reshape.
#         C = size(x, 1)
#         x = reshape(x[i], C, 1, 1)
#     elseif isequal(N, 2)
#         C = size(x, 1)
#         B = size(x, 2)
#         x[i] = reshape(x[i], C, 1, B)
#     end
#     return nothing
# end


function paramsof(att::FusedAttention)
    Q = att.Qf
    K = att.Kf
    V = att.Vf
    params = Vector{Variable}()
    append!(params, paramsof(Q.inner))
    append!(params, paramsof(Q.outer))
    append!(params, paramsof(K.inner))
    append!(params, paramsof(K.outer))
    append!(params, paramsof(V.inner))
    append!(params, paramsof(V.outer))
    return params
end


function xparamsof(att::FusedAttention)
    Q = att.Qf
    K = att.Kf
    V = att.Vf
    params = Vector{XVariable}()
    append!(params, xparamsof(Q.inner))
    append!(params, xparamsof(Q.outer))
    append!(params, xparamsof(K.inner))
    append!(params, xparamsof(K.outer))
    append!(params, xparamsof(V.inner))
    append!(params, xparamsof(V.outer))
    return params
end

function nparamsof(att::FusedAttention)
    Q = att.Qf
    K = att.Kf
    V = att.Vf
    n = 0
    n += nparamsof(Q.inner)
    n += nparamsof(Q.outer)
    n += nparamsof(K.inner)
    n += nparamsof(K.outer)
    n += nparamsof(V.inner)
    n += nparamsof(V.outer)
    return n
end



function elsizeof(att::FusedAttention)
    Q = att.Qf
    K = att.Kf
    V = att.Vf
    n = elsizeof(Q.inner)
    !isnothing(n) && return n
    n = elsizeof(Q.outer)
    !isnothing(n) && return n
    n = elsizeof(K.inner)
    !isnothing(n) && return n
    n = elsizeof(K.outer)
    !isnothing(n) && return n
    n = elsizeof(V.inner)
    !isnothing(n) && return n
    n = elsizeof(V.outer)
    !isnothing(n) && return n
    @warn "returning a fake element size 4. You should pay attention"
    return 4
end


function bytesof(att::FusedAttention, unit::String="MB")
    n = nparamsof(att) * elsizeof(att)
    return blocksize(n, uppercase(unit))
end
