export backward


function backward(topnode::Variable{T}, d::Union{Real,T}=1.0f0; keepgraph::Bool=false, by::String="dfs") where T
    if need2computeδ!(topnode)
        if d isa Real
            δ(topnode) .= eltype(T)(d)
        else
            δ(topnode) .= d
        end
    end

    if by=="dfs"
        sorted = sort_by_dfs(topnode)
    end
    if by=="bfs"
        sorted = sort_by_bfs(topnode)
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
