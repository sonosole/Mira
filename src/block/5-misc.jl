kbytesof(b::B) where B <: Block = bytesof(b, "KB")
mbytesof(b::B) where B <: Block = bytesof(b, "MB")
gbytesof(b::B) where B <: Block = bytesof(b, "GB")
tbytesof(b::B) where B <: Block = bytesof(b, "TB")


function bytesof(blocks::Vector, unit::String="kb")
    n = 0
    for b in blocks
        if b isa Block
            n += bytesof(b, unit)
        end
    end
    return n
end


function kbytesof(blocks::Vector)
    n = 0
    for b in blocks
        if b isa Block
            n += kbytesof(b)
        end
    end
    return n
end


function mbytesof(blocks::Vector)
    n = 0
    for b in blocks
        if b isa Block
            n += mbytesof(b)
        end
    end
    return n
end


function gbytesof(blocks::Vector)
    n = 0
    for b in blocks
        if b isa Block
            n += gbytesof(b)
        end
    end
    return n
end


function tbytesof(blocks::Vector)
    n = 0
    for b in blocks
        if b isa Block
            n += tbytesof(b)
        end
    end
    return n
end


function paramsof(blocks::Vector)
    params = Vector{Variable}(undef,0)
    for b in blocks
        if b isa Block
            p = paramsof(b)
            if p ≠ nothing
                append!(params, p)
            end
        end
    end
    return params
end


function xparamsof(blocks::Vector)
    xparams = Vector{XVariable}(undef,0)
    for b in blocks
        if b isa Block
            p = xparamsof(b)
            if p ≠ nothing
                append!(xparams, p)
            end
        end
    end
    return xparams
end


function nparamsof(blocks::Vector)
    n = 0
    for b in blocks
        if b isa Block
            n += nparamsof(b)
        end
    end
    return n
end



function paramsof(f::Function)
    return nothing
end


function xparamsof(f::Function)
    return nothing
end


function nparamsof(f::Function)
    return 0
end


function nops(f::Function, c::Int=1)
    return (0, 0, 0) .* c
end

# const FC = Union{Affine,
#                  Linear,
#                  Dense,
#                  Maxout,
#                  Res0d,
#                  Res0dWithBN,
#                  SelfLoopResNet,
#                  SelfLoopCumulativeResNet,
#                  MeanNormResDense
# }

# export Keep3dimsForward
# export Keep3dimsPredict
# export KeepNdimsForward
# export KeepNdimsPredict

# function Keep3dimsForward(block::FC, x::Variable)
#     C,T,B = size(x)
#     y = reshape(x, (C, T*B))  # 3D --> 2D
#     z = forward(block, y)
#     C, TB = size(z)
#     return reshape(z, (C,T,B))  # 2D --> 3D
# end

# function Keep3dimsPredict(block::FC, x::AbstractArray)
#     C,T,B = size(x)
#     y = reshape(x, (C, T*B))  # 3D --> 2D
#     z = predict(block, y)
#     C, TB = size(z)
#     return reshape(z, (C,T,B))  # 2D --> 3D
# end



# function KeepNdimsForward(block::FC, x::Variable)
#     S = size(x)             # N-Dim Variable
#     F = S[1]                # Feat-Dim
#     B = prod(S[2:end])      # Batch-Dim
#     y = reshape(x, (F, B))  # N-D to 2D
#     z = forward(block, y)   # forward once

#     F, B = size(z)          # Feat-Dim, Batch-Dim
#     V = ntuple(i -> i≠1 ? S[i] : F, length(S))
#     return reshape(z, V)    # 2D to N-D
# end

# function KeepNdimsPredict(block::FC, x::Variable)
#     S = size(x)             # N-Dim Variable
#     F = S[1]                # Feat-Dim
#     B = prod(S[2:end])      # Batch-Dim
#     y = reshape(x, (F, B))  # N-D to 2D
#     z = predict(block, y)   # forward once

#     F, B = size(z)          # Feat-Dim, Batch-Dim
#     V = ntuple(i -> i≠1 ? S[i] : F, length(S))
#     return reshape(z, V)    # 2D to N-D
# end


"""
    checkvalues(x::AbstractArray)
If any value in x is NaN or Inf, throw a info
"""
function checkvalues(x::AbstractArray)
    for v in Array(x)
        if isnan(v) || isinf(v)
            @info red!("$v is unnormal")
            return nothing
        end
    end
end


function checkvalues(cv::Vector{XVariable})
    for (c, v) in cv
        checkvalues(value(v))
    end
end


function checkvalues(vs::Vector{Variable})
    for v in vs
        checkvalues(value(v))
    end
end


"""
    staticsof(x::AbstractArray) -> mean, std, min, max
"""
function staticsof(x::AbstractArray)
    μ = mean(x)
    σ = std(x, mean=μ)
    return μ, σ, minimum(x), maximum(x)
end


"""
    staticsof(cv::Vector{XVariable})
Show mean, std, min, max
"""
function staticsof(cv::Vector{XVariable})
    for (i, (c, v)) in enumerate(cv)
        μ, σ, l, u = staticsof(value(v))
        println("$(i)\t$(size(v))\t$c\t[$l, $u]\t($μ ± $σ)")
    end
end


"""
    staticsof(vs::Vector{Variable})
Show mean, std, min, max
"""
function staticsof(vs::Vector{Variable})
    for (i, v) in enumerate(vs)
        μ, σ, l, u = staticsof(value(v))
        println("$(i)\t$(size(v))\t[$l, $u]\t($μ\t±\t$σ)")
    end
end


forward(f::Function, x::Variable)      = f(x)
predict(f::Function, x::AbstractArray) = f(x)

function clone(f::Function; type::Type=Array{Float32})
    return f
end
