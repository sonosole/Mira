export quant_mseloss

function quant_mseloss(x::AbstractArray, Qmin::Real, Qmax::Real; n::Int=32)
    Xmin = minimum(x)
    Xmax = maximum(x)
    N = 1/length(x)
    m = (Xmin + Xmax) / 2     # middle value
    Δ = (Xmax - Xmin) / n # delta
    minloss = 1/N
    minx = Xmin
    maxx = Xmax
    L = zeros(n)
    for i = 1:n
        r = i * Δ
        y = xqx.(x, m-r, m+r, Qmin, Qmax)
        L[i] = sum(abs.(x .- y)) * N + 1e-9
        if L[i] < minloss
            minloss = L[i]
            minx = m - r
            maxx = m + r
        end
    end
    return L, minx, maxx
end
