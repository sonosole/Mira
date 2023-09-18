export backward
export backprop

function try_to_free_grad(x::Variable)
    if !x.keepsgrad
        x.delta = nothing
    end
end

function free_data_vjp_kids(x::Variable)
    x.value    = nothing
    x.backward = nothing
    x.children = nothing
end


function backward(y::Variable{T},
                  g::Union{Real,T}=1.0f0;
                  partial::Bool=false,
                  keepgraph::Bool=false,
                  by::String="dfs") where T

    filldelta(y, g)

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
            try_to_free_grad(v)
            free_data_vjp_kids(v)
        end
    else
        for v in sorted
            v.backward()
        end
    end
end



function backprop(sorted::Vector{Variable})
    for node in sorted
        node.backward()
    end
end
