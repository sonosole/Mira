export minmax_quant_loss
export center_quant_loss


"""
    minmax_quant_loss(x::AbstractArray, Qmin::Real, Qmax::Real; n::Int=32, c::Int=2)
# Example
    using Plots
    x1 = 0.9randn(1024*1024) .- 3.0;
    x2 = 0.5randn(1024*1024) .+ 2.0;
    x  = [x1; x2];
    dist, lower, upper = center_quant_loss(x, -127, 127, n=32, c=4);
    plot(histogram(x), plot(c), layout=(2,1))
"""
function minmax_quant_loss(x::AbstractArray, Qmin::Real, Qmax::Real; n::Int=32, c::Int=2)
    Xmin = minimum(x)
    Xmax = maximum(x)
    N = 1 / length(x)
    m = (Xmin + Xmax) / 2         # middle value
    Δ = (Xmax - Xmin) / (2*n-1)   # divided in 2*n pieces

    # init values
    minloss = Inf
    minx = Xmin
    maxx = Xmax
    L = zeros(c*n)

    for i = 1:c*n
        δ = i * Δ
        y = xqx(x, m - δ, m + δ, Qmin, Qmax)
        L[i] = sum(abs.(x .- y)) * N + 1e-38
        if L[i] < minloss
            minloss = L[i]
            minx = m - δ
            maxx = m + δ
        end
    end
    return log.(L), minx, maxx
end


"""
    center_quant_loss(x::AbstractArray, Qmin::Real, Qmax::Real; n::Int=32, c::Int=2)
# Example
    using Plots
    x1 = 0.7randn(1024*1024) .- 3.0;
    x2 = 0.5randn(1024*1024) .+ 2.0;
    x  = [x1; x2];
    dist, lower, upper = center_quant_loss(x, -127, 127, n=32, c=4);
    plot(histogram(x), plot(c), layout=(2,1))
"""
function center_quant_loss(x::AbstractArray, Qmin::Real, Qmax::Real; n::Int=32, c::Int=2)
    Xmin = minimum(x)
    Xmax = maximum(x)
    N = 1 / length(x)
    μ = mean(x)     # middle value
    R = Xmax - μ    # Right length
    L = Xmin - μ    # Left length

    Δ = max(L, R) / (n-1)   # divided in 2*n pieces

    # init values
    minloss = Inf
    minx = Xmin
    maxx = Xmax
    L = zeros(c*n)

    for i = 1:c*n
        δ = i * Δ
        y = xqx(x, μ - δ, μ + δ, Qmin, Qmax)
        L[i] = sum(abs.(x .- y)) * N + 1e-38
        if L[i] < minloss
            minloss = L[i]
            minx = μ - δ
            maxx = μ + δ
        end
    end
    return log.(L), minx, maxx
end
