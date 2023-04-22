export backward


function backward(y::Variable{T},
                  δy::Union{Real,T}=1.0f0;
                  partial::Bool=false,
                  keepgraph::Bool=false,
                  by::String="dfs") where T

    if need2computeδ!(y)
        if δy isa Real
            δ(y) .= eltype(T)(δy)
        else
            δ(y) .= δy
        end
    end

    # partial==true means y is one of the loss functions
    partial && resetindegree(y)

    if by=="dfs"
        sorted = sort_by_dfs(y)
    end
    if by=="bfs"
        sorted = sort_by_bfs(y)
    end

    if !keepgraph
        for v in sorted
            v.backward()
            v = nothing
        end
    else
        for v in sorted
            v.backward()
        end
    end
end


export backprop
function backprop(sorted::Vector{Variable})
    for node in sorted
        node.backward()
    end
end


function backprop(sorted::Vector{Variable}, dest::Variable)
    for node in sorted
        node.backward()
        dest.indegree == 0 && break
    end
end
