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
