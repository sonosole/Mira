export freeze
export freezed
export unfreeze
export unfreezed


"""
    freeze(x::Union{Variable, Variables, XVariable, XVariables})
Freeze the params, so they could NOT get involved into training.
"""
function freeze(x::Variable)
    x.keepsgrad = false
    return nothing
end

function freeze(x::XVariable)
    return freeze(last(x))
end

function freeze(vs::Variables)
    for v in vs
        freeze(v)
    end
end

function freeze(xs::XVariables)
    for x in xs
        freeze(x)
    end
end


"""
    unfreeze(x::Union{Variable, Variables, XVariable, XVariables})
Unfreeze the params, so they could get involved into training.
"""
function unfreeze(x::Variable)
    x.keepsgrad = true
    return nothing
end

function unfreeze(x::XVariable)
    return unfreeze(last(x))
end

function unfreeze(vs::Variables)
    for v in vs
        unfreeze(v)
    end
end

function unfreeze(xs::XVariables)
    for x in xs
        unfreeze(x)
    end
end


"""
    freezed(::XVariables)::XVariables
Return the freezed XVariables. This is usually used with `xparamsof`.

# Example
```julia
net = Dense(2,3);
freeze(net.w);                      # freeze the weigths of net
frozen = freezed(xparamsof(net))    # only the weigths of net is kept
```
"""
function freezed(xparams::XVariables)
    valid = VecXVariable(0)
    for p in xparams
        if !keepsgrad(last(p))
            push!(valid, p)
        end
    end
    return valid
end


"""
    freezed(::Variables)::Variables
Return the freezed Variables. This is usually used with `paramsof`.

# Example
```julia
net = Dense(2,3);
freeze(net.w);                     # freeze the weigths of net
frozen = freezed(paramsof(net))    # only the weigths of net is kept
```
"""
function freezed(params::Variables)
    valid = VecVariable(0)
    for p in params
        if !keepsgrad(p)
            push!(valid, p)
        end
    end
    return valid
end



"""
    unfreezed(::XVariables)::XVariables
Return the unfreezed XVariables. This is usually used with `xparamsof`.

# Example
```julia
net = Dense(2,3);
freeze(net.w);                        # freeze the weigths of net
learnable = unfreezed(xparamsof(net)) # only the bias of net is kept
```
"""
function unfreezed(xparams::XVariables)
    valid = VecXVariable(0)
    for p in params
        if keepsgrad(last(p))
            push!(valid, p)
        end
    end
    return valid
end


"""
    unfreezed(::Variables)::Variables
Return the unfreezed Variables. This is usually used with `paramsof`.

# Example
```julia
net = Dense(2,3);
freeze(net.w);                       # freeze the weigths of net
learnable = unfreezed(paramsof(net)) # only the bias of net is kept
```
"""
function unfreezed(params::Variables)
    valid = VecVariable(0)
    for p in params
        if keepsgrad(p)
            push!(valid, p)
        end
    end
    return valid
end
