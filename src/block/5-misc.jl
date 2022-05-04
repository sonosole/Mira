kbytesof(b::B) where B <: Block = bytesof(b, "KB")
mbytesof(b::B) where B <: Block = bytesof(b, "MB")
gbytesof(b::B) where B <: Block = bytesof(b, "GB")
tbytesof(b::B) where B <: Block = bytesof(b, "TB")


const FC = Union{Affine,
                 Linear,
                 Dense,
                 Maxout,
                 Res0d,
                 Res0dWithBN,
                 SelfLoopResNet,
                 SelfLoopCumulativeResNet,
                 MeanNormResDense
}

export Keep3dimsForward
export Keep3dimsPredict
export KeepNdimsForward
export KeepNdimsPredict

function Keep3dimsForward(block::FC, x::Variable)
    C,T,B = size(x)
    y = reshape(x, (C, T*B))  # 3D --> 2D
    z = forward(block, y)
    C, TB = size(z)
    return reshape(z, (C,T,B))  # 2D --> 3D
end

function Keep3dimsPredict(block::FC, x::AbstractArray)
    C,T,B = size(x)
    y = reshape(x, (C, T*B))  # 3D --> 2D
    z = predict(block, y)
    C, TB = size(z)
    return reshape(z, (C,T,B))  # 2D --> 3D
end



function KeepNdimsForward(block::FC, x::Variable)
    S = size(x)             # N-Dim Variable
    F = S[1]                # Feat-Dim
    B = prod(S[2:end])      # Batch-Dim
    y = reshape(x, (F, B))  # N-D to 2D
    z = forward(block, y)   # forward once

    F, B = size(z)          # Feat-Dim, Batch-Dim
    V = ntuple(i -> i≠1 ? S[i] : F, length(S))
    return reshape(z, V)    # 2D to N-D
end

function KeepNdimsPredict(block::FC, x::Variable)
    S = size(x)             # N-Dim Variable
    F = S[1]                # Feat-Dim
    B = prod(S[2:end])      # Batch-Dim
    y = reshape(x, (F, B))  # N-D to 2D
    z = predict(block, y)   # forward once

    F, B = size(z)          # Feat-Dim, Batch-Dim
    V = ntuple(i -> i≠1 ? S[i] : F, length(S))
    return reshape(z, V)    # 2D to N-D
end
