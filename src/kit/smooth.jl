export Conv1dSmoother
export smooth

"""
    Conv1dSmoother
a smoothing operator for 1-d data
# Example
Conv1dSmoother(smoother::AbstractArray=[0.2, 0.6, 0.2]; type::Type=Array{Float32})
"""
struct Conv1dSmoother
    w::AbstractArray # smooth weights
    k::Int           # kernel size
    function Conv1dSmoother(smoother::AbstractArray=[0.2, 0.6, 0.2]; type::Type=Array{Float32})
        k = length(smoother)
        @assert mod(k,2)==1 "length of smoother should be a odd number"
        w = reshape(smoother ./ sum(smoother), 1, k)
        new(type(w), k)
    end
end


# pretty printing
function Base.show(io::IO, x::Conv1dSmoother) where T
    println(cyan!("═══ Conv1dSmoother ═══"))
    display(x.w)
end


function smooth(smoother::Conv1dSmoother, x::S) where S <: AbstractArray
    @assert ndims(x)==3 "input shape is of (ichannels, width, batchsize)"
    channels, T, batchsize = size(x)    # T is timesteps
    K = smoother.k
    w = smoother.w
    y = zero(x)
    Δ = div(K, 2)
    l = one(eltype(x))  # alias for one

    # left side smoothing
    Threads.@threads for t = 1:Δ
        L = (Δ+1)-(t-1) # left starting point
        A = l ./ sum(w[:,L:K], dims=2)
        y[:,t:t,:] = sum(w[:, L:K] .* x[:, 1:Δ+t, :], dims=2) .* A
    end

    # middle part smoothing
    Threads.@threads for t = Δ+1 : T-Δ
        y[:,t:t,:] = sum(w .* x[:, t-Δ : t+Δ, :], dims=2)
    end

    # right side smoothing
    Threads.@threads for t = T-Δ+1 : T
        R = T-t+Δ+1 # right stopping point
        A = l ./ sum(w[:,1:R], dims=2)
        y[:,t:t,:] = sum(w[:,1:R] .* x[:, t-Δ:T, :], dims=2) .* A
    end
    return y
end
