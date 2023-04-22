export lrarray

"""
    lrarray(init,final,steps;func="exp") -> Vector{Float32}
# Arguments
- ` init`: initial learning rate
- `final`: final learning rate
- `steps`: steps to change from `init` to `final`
- ` func`: function used to change the learning rate. (e.g. exp/cos/linear)
"""
function lrarray(init,final,steps;func="exp")
    func=="exp"    && return lrarrayexp(init,final,steps)
    func=="cos"    && return lrarraycos(init,final,steps)
    func=="linear" && return lrarraylinear(init,final,steps)
end


function lrarrayexp(i,f,n)
    if n == 1
        return [i]
    else
        lr = zeros(Float32,n)
        α  = log(f/i)/(n-1)
        for x = 1:n
            lr[x] = i * exp(α * (x-1))
        end
        return lr
    end
end

function lrarraylinear(i,f,n)
    if n == 1
        return [i]
    else
        lr = zeros(Float32,n)
        α  = (f-i)/(n-1)
        for x = 1:n
            lr[x] = α*(x-1) + i
        end
        return lr
    end
end


function lrarraycos(i,f,n)
    if n == 1
        return [i]
    else
        lr = zeros(Float32,n)
        α  = pi/2/(n-1)
        Δ  = (i-f)
        for x = 1:n
            lr[x] = cos(α*(x-1)) * Δ + f
        end
        return lr
    end
end
