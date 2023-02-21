export zerodelta
export clone
export need2computeδ!
export ifNotKeepδThenFreeδ!
export elsizeof
export value, delta, ᵛ, ᵟ, δ
export isleaf, setleaf, backprop, keepsgrad, needsgrad
export haschild, childrenof, addchild, nchildrenof
export haskid, kidsof, addkid, nkidsof

export visited
export setvisited
export unsetvisited
export VecVariable
export VecXVariable

export Variable, Variables
export XVariable, XVariables
export VarOrNil


"""
    mutable struct Variable{T} where T <: AbstractArray

# Constructor
    function Variable{T}(x,
                         backprop  :: Bool=true,  # when forward, it's true
                         keepsgrad :: Bool=false, # whether keeps grad after backward
                         isleaf    :: Bool=false) # whether leaf node
"""
mutable struct Variable{T}
    value     :: T                   # value in forward
    delta     :: Union{Nothing,T}    # gradients collected in backprop
    shape     :: Tuple               # shape of `value`
    isleaf    :: Bool                # whether leaf node
    backprop  :: Bool                # whether needs backprop when forward
    keepsgrad :: Bool                # whether keeps grad after backprop
    visited   :: Bool                # whether visited during backprop
    backward  :: FunOrNil            # backward function
    children  :: Vector{Variable{T}} # children Variables
    function Variable{T}(x, backprop  :: Bool=true,
                            keepsgrad :: Bool=false,
                            isleaf    :: Bool=false) where T <: AbstractArray
        delta    = nothing
        shape    = size(x)
        visited  = false
        backward = nothing
        children = Vector{Variable{T}}()
        new{T}(x, delta, shape, isleaf, backprop, keepsgrad, visited, backward, children)
    end
end


# Convenient type-specilized constructors for data on GPU/CPU/xPU etc....
function Variable(x; backprop::Bool=true,
                     keepsgrad::Bool=false,
                     type::Type=Array{Float32})
    isleaf = true    # any user defined Variable is a leaf
    return Variable{type}(x, backprop, keepsgrad, isleaf)
end


const XVariable  = Tuple{Char, Variable}
const VarOrNil   = Union{Variable, Nothing}
const Variables  = Vector{Variable}
const XVariables = Vector{XVariable}

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", x::Variable)
    if  x.isleaf println(cyan!("\n═══ Leaf Variable ═══")) end
    if !x.isleaf println(cyan!("\n═══ None Leaf Variable ═══")) end

    print(yellow!("\nvalue is "))
    display(x.value)
    print(green!("\ndelta is "))
    display(x.delta)
end


function clone(x::Variable; type::Type=Array{Float32})
    return Variable{type}(x.value, x.backprop, x.keepsgrad, x.isleaf)
end


function clone(x::Nothing; type::Type=Array{Float32})
    return nothing
end


function zerodelta(x::Variable{T}) where T
    if isnothing(x.delta)
        x.delta = Zeros(T, x.shape);
    end
end


function need2computeδ!(x::Variable)
    # 1. 不需要学习的叶子参数不需要初始化，其他情况都要。
    # 2. 当某叶子节点的 keepsgrad==false 时，则此叶子节
    #   点不参与反向传播的计算，也即达到了冻结参数的目的
    if !(x.isleaf && !x.keepsgrad)
        zerodelta(x)
        return true
    else
        return false
    end
end


function ifNotKeepδThenFreeδ!(x::Variable)
    if !x.keepsgrad
        x.delta = nothing
    end
end



Base.sizeof(x::Variable)         =  sizeof(x.value)
Base.size(x::Variable)           =    size(x.value)
Base.size(x::Variable, dim::Int) =    size(x.value, dim)
Base.ndims(x::Variable)          =   ndims(x.value)
Base.length(x::Variable)         =  length(x.value)
Base.strides(x::Variable)        = strides(x.value)
Base.eltype(x::Variable)         =  eltype(x.value)
Base.similar(x::Variable{T})  where T = Variable{T}( similar(x.value), x.backprop, x.keepsgrad, x.isleaf)
Base.copy(x::Variable{T})     where T = Variable{T}(    copy(x.value), x.backprop, x.keepsgrad, x.isleaf)
Base.deepcopy(x::Variable{T}) where T = Variable{T}(deepcopy(x.value), x.backprop, x.keepsgrad, x.isleaf)

Base.setindex!(x::Variable, v::Number,        k...) = (x.value[k...] .= v)
Base.setindex!(x::Variable, v::AbstractArray, k...) = (x.value[k...]  = v)



function Base.getindex(x::Variable{T}, k...) where T
    !x.backprop && return x.value[k...]
    y = Variable{T}(x.value[k...], x.backprop, x.keepsgrad, x.isleaf)
    if y.backprop
        y.backward = function ∇getindex()
            if need2computeδ!(x)
                x.delta[k...] .+= y.delta
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function Base.getindex(x::Variable{T}, k::Int) where T
    !x.backprop && return x.value[k:k]
    y = Variable{T}(x.value[k:k], x.backprop, x.keepsgrad, x.isleaf)
    if y.backprop
        y.backward = function ∇getindex()
            if need2computeδ!(x)
                x.delta[k:k] .+= y.delta
            end
            ifNotKeepδThenFreeδ!(y)
        end
        addchild(y, x)
    end
    return y
end


function (v::Variable)(i...)
    if v.delta ≠ nothing
        return v.delta[i...]
    else
        return nothing
    end
end



# pretty printing
function Base.show(io::IO, ::MIME"text/plain", xv::XVariable)
    c, x = xv
    if  x.isleaf println(cyan!("\n═══ Leaf Variable ($c) ═══")) end
    if !x.isleaf println(cyan!("\n═══ None Leaf Variable ═══")) end

    print(yellow!("\nvalue is "))
    display(x.value)
    print(green!("\ndelta is "))
    display(x.delta)
end


elsizeof(x::Variable) = sizeof(eltype(x))


# lazy showing way of Variable's main vars
@inline ᵛ(x::Variable) = x.value
@inline ᵟ(x::Variable) = x.delta
@inline δ(x::Variable) = x.delta
@inline value(x::Variable) = x.value
@inline delta(x::Variable) = x.delta

# Variable's states fns
@inline isleaf(x::Variable) = x.isleaf
@inline backprop(x::Variable) = x.backprop
@inline keepsgrad(x::Variable) = x.keepsgrad
@inline needsgrad(x::Variable) = x.keepsgrad = true
@inline haschild(x::Variable) = length(x.children) > 0 ? true : false
@inline   haskid(x::Variable) = length(x.children) > 0 ? true : false
@inline childrenof(x::Variable) = x.children
@inline     kidsof(x::Variable) = x.children
@inline nchildrenof(x::Variable) = length(x.children)
@inline     nkidsof(x::Variable) = length(x.children)

@inline visited(x::Variable)      = x.visited
@inline setvisited(x::Variable)   = x.visited = true
@inline unsetvisited(x::Variable) = x.visited = false


@inline addchild(p::Variable, c::Variable) = !c.isleaf && push!(p.children, c)
@inline   addkid(p::Variable, c::Variable) = !c.isleaf && push!(p.children, c)


function VecVariable(n::Int=0)
    return Vector{Variable}(undef, n)
end

function VecXVariable(n::Int=0)
    return Vector{XVariable}(undef, n)
end


function setleaf(x::Variable)
    x.isleaf = true
    return nothing
end


# having the below defines, the activation function
# could be set nothing for blocks like Dense Conv
(f::Nothing)(x::AbstractArray) = x
(f::Nothing)(x::VarOrNil) = x
