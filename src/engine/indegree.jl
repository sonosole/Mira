export indegree
export indegree_by_bfs
export indegree_by_dfs


function indegree_by_bfs(topnode::Variable)
    idegree = Dict{Variable,Int}()
    queue = []
    push!(queue, topnode)
    setvisited(topnode)
    while length(queue) ≠ 0
        node = popfirst!(queue)
        if haskid(node)
            for kid in kidsof(node)
                if !visited(kid)
                    push!(queue, kid)  # once pushed,
                    setvisited(kid)    # set visited.
                end
            end
        end
        # all zero init
        push!(idegree, node => 0)
    end

    for (node, ins) in idegree
        unsetvisited(node)
    end
    return idegree
end


function indegree_by_dfs(topnode::Variable)
    idegree = Dict{Variable,Int}()
    stack = []
    push!(stack, topnode)
    setvisited(topnode)
    while length(stack) ≠ 0
        node = pop!(stack)
        if haskid(node)
            for kid in kidsof(node)
                if !visited(kid)
                    push!(stack, kid)  # once pushed,
                    setvisited(kid)    # set visited.
                end
            end
        end
        # all zero init
        push!(idegree, node => 0)
    end

    for (node, ins) in idegree
        unsetvisited(node)
    end
    return idegree
end


function indegree(topnode::Variable; by::Function=indegree_by_dfs)
    idegree = by(topnode)
    # calculate indegree of each node
    for (node, ins) in idegree
        if haskid(node)
            for kid in kidsof(node)
                idegree[kid] += 1
            end
        end
    end
    return idegree
end
