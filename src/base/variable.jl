export zerodelta
export clone
export need2computeδ!
export ifNotKeepδThenFreeδ!
export elsizeof
export value, delta, ᵛ, ᵟ, δ
export isleaf, backprop, keepsgrad
export haschild, childrenof, addchild, nchildrenof
export haskid, kidsof, addkid, nkidsof

export visited
export setvisited
export unsetvisited
export Vecvar

export Variable, Variables
export XVariable, XVariables
export VarOrNil, FunOrNil
const FunOrNil = Union{Function, Nothing}




"""
    mutable struct Variable{T}

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
const XVariables = Vector{Tuple{Char, Variable}}

# pretty printing
function Base.show(io::IO, x::Variable{T}) where T
    if  x.isleaf println(cyan!("\n═══ Leaf Variable ═══")) end
    if !x.isleaf println(cyan!("\n═══ None Leaf Variable ═══")) end

    print(blue!("\nvalue is "))
    display(x.value)
    print(green!("\ndelta is "))
    display(x.delta)
end


function clone(x::Variable; type::Type=Array{Float32})
    return Variable{type}(x.value, x.backprop, x.keepsgrad, x.isleaf)
end


function zerodelta(x::Variable{T}) where T
    if isnothing(x.delta)
        x.delta = Zeros(T, x.shape);
    end
end


function need2computeδ!(x::Variable{T}) where T
    # 不需要学习的叶子参数不需要初始化，其他情况都要
    if !(x.isleaf && !x.keepsgrad)
        zerodelta(x)
        return true
    else
        return false
    end
end


function ifNotKeepδThenFreeδ!(x::Variable{T}) where T
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
    y = Variable{T}(x.value[k...], x.backprop, x.keepsgrad, x.isleaf)
    if y.backprop
        y.backward = function getindexBackward()
            if need2computeδ!(x)
                x.delta[k...] .+= y.delta
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
function Base.show(io::IO, xv::XVariable)
    c, x = xv
    if  x.isleaf println(cyan!("\n═══ Leaf Variable ($c) ═══")) end
    if !x.isleaf println(cyan!("\n═══ None Leaf Variable ═══")) end

    print(blue!("\nvalue is "))
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

function Vecvar(n::Int=0)
    return Vector{Variable}(undef, n)
end
