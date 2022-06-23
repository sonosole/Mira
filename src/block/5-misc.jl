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
