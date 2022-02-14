export backward!
export backward_by_dfs
export backward_by_indegree


function backward!(x::Variable{T}, d::Union{Nothing,T}=nothing; keepgraph::Bool=false, by::String="dfs") where T
    if !isnothing(d) δ(x) .= d end
    by=="dfs" && return backward_by_dfs(x, keepgraph)
    by=="indegree" && return backward_by_indegree(x, keepgraph)
    @error "by is dfs or indegree, but got $by"
end



function backward_by_dfs(topnode::Variable{T}, keepgraph::Bool=false) where T
    sorted = sort_by_dfs(topnode)
    if !keepgraph
        for v in sorted
            v.backward()
            v = nothing
        end
    else
        for v in sorted
            v.backward()
            unsetvisited(v)
        end
    end
end


function backward_by_indegree(topnode::Variable, keepgraph::Bool=false)
    sorted = sort_by_indegree(topnode)
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
