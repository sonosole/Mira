export dropout
export dropout!
export xdropout
export xdropout!


"""
    dropout(x::Variable{Array{T}}; p::Real=0.1f0) -> y::Variable{Array{T}}

Randomly zeros some elements of the input tensor `x` with probability `p` using samples
from a Bernoulli distribution. This is an effective way to regularization and preventing
the co-adaptation of neurons. The output elements of `y` are scaled by a factor of `1/(1-p)`
during training. `dropout` should be removed at evaluation. Dropout is also viewed as a
mean of data augmentation.
"""
function dropout(x::Variable{Array{T}}; p::Real=0.1f0) where T
    @assert 0.0≤p<1.0 "p is in [0,1), but got p=$p"
    l = T(1)
    p = T(p)
    m = (rand(T, x.shape) .< (l - p)) .* (l/(l - p)) # weighted mask
    y = Variable{Array{T}}(ᵛ(x) .* m, x.backprop)
    if y.backprop
        y.backward = function ∇dropout()
            if need2computeδ!(x)
                x ← δ(y) .* m
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end



"""
    dropout!(x::Variable{Array{T}}; p::Real=0.1f0) -> y::Variable{Array{T}}

Randomly zeros some elements of the input tensor `x` with probability `p` using samples
from a Bernoulli distribution. This is an effective way to regularization and preventing
the co-adaptation of neurons. The output elements of `y` are scaled by a factor of `1/(1-p)`
during training. `dropout!` should be removed at evaluation. Dropout is also viewed as a
mean of data augmentation.
"""
function dropout!(x::Variable{Array{T}}; p::Real=0.1f0) where T
    @assert 0.0≤p<1.0 "p is in [0,1), but got p=$p"
    l = T(1)
    p = T(p)
    m = (rand(T, x.shape) .< (l - p)) .* (l/(l - p)) # weighted mask
    y = Variable{Array{T}}(dotmul!(ᵛ(x), m), x.backprop)
    if y.backprop
        y.backward = function ∇dropout!()
            if need2computeδ!(x)
                x ← δ(y) .* m
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    xdropout(x::Variable{Array{T}}; p::Real=0.1f0, dims::IntOrDims{D}=1) -> y::Variable{Array{T}}

Randomly zeros some `slices` of the input tensor `x` with probability `p` using samples
from a Bernoulli distribution. This is an effective way to regularization and preventing
the co-adaptation of neurons. The output elements of `y` are scaled by a factor of `1/(1-p)`
during training. `xdropout` should be removed at evaluation. Dropout is also viewed as a
mean of data augmentation.
"""
function xdropout(x::Variable{Array{T}}; p::Real=0.1f0, dims::IntOrDims{D}=1) where {T,D}
    @assert 0.0≤p<1.0 "p is in [0,1), but got p=$p"
    l = T(1)
    p = T(p)
    S = dimsfilter(size(x), dims)
    m = (rand(T, S) .< (l - p)) .* (l/(l - p))  # weighted mask
    y = Variable{Array{T}}(ᵛ(x) .* m, x.backprop)
    if x.backprop
        y.backward = function ∇xdropout()
            if need2computeδ!(x)
                x ← δ(y) .* m
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


"""
    xdropout!(x::Variable{Array{T}}; p::Real=0.1f0, dim::IntOrDims{D}=1) -> y::Variable{Array{T}}

Randomly zeros some `slices` of the input tensor `x` with probability `p` using samples
from a Bernoulli distribution. This is an effective way to regularization and preventing
the co-adaptation of neurons. The output elements of `y` are scaled by a factor of `1/(1-p)`
during training. `xdropout!` should be removed at evaluation. Dropout is also viewed as a
mean of data augmentation.
"""
function xdropout!(x::Variable{Array{T}}; p::Real=0.1f0, dim::IntOrDims{D}=1) where {T,D}
    @assert 0.0≤p<1.0 "p is in [0,1), but got p=$p"
    l = T(1)
    p = T(p)
    S = dimsfilter(size(x), dims)
    m = (rand(T, S) .< (l - p)) .* (l/(l - p))  # weighted mask
    y = Variable{Array{T}}(dotmul!(ᵛ(x), m), x.backprop)
    if y.backprop
        y.backward = function ∇xdropout!()
            if need2computeδ!(x)
                x ← δ(y) .* m
            end
            ifNotKeepδThenFreeδ!(y);
        end
        addchild(y, x)
    end
    return y
end
