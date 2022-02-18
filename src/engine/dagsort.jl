export sort_by_recursive_dfs
export sort_by_dfs
export sort_by_indegree

"""
    sort_by_recursive_dfs(topnode::Variable) -> stack
"""
function sort_by_recursive_dfs(topnode::Variable)
    stack = Vector{Variable}()
    function visit(node::Variable)
        setvisited(node)
        if haskid(node)
            for kid in kidsof(node)
                if !visited(kid)
                    visit(kid)
                end
            end
        end
        push!(stack, node)
    end
    visit(topnode)
    return stack
end


"""
    sort_by_dfs(topnode::Variable) -> stack

nonrecursive dfs sort
"""
function sort_by_dfs(topnode::Variable)
    store = Vector{Variable}()
    stack = Vector{Variable}()

    function visit(node::Variable)
        setvisited(node)
        for kid in kidsof(node)
            if !visited(kid)
                push!(stack, kid)
            end
        end
    end

    if !visited(topnode)
        push!(stack, topnode)
        lastnode = stack[end]
    else
        @warn "input has been visited"
        return nothing
    end

    while length(stack)≠0
        while haskid(lastnode)
            len = length(stack)
            visit(lastnode)
            if length(stack) ≠ len
                lastnode = stack[end]
            else  # node has been visited
                break
            end
        end
         # the last node (leaf node) has no
         # kids but should be marked visited
        setvisited(lastnode)

        if stack≠[]
            push!(store, pop!(stack))
        end
        if length(stack)≠0
            lastnode = stack[end]
        end

        while visited(lastnode)
            if stack≠[]
                push!(store, pop!(stack))
            end
            if length(stack)≠0
                lastnode = stack[end]
            else
                break
            end
        end
    end
    return store
end


"""
    sort_by_indegree(topnode::Variable) -> queue
"""
function sort_by_indegree(topnode::Variable; by::Function=indegree_by_bfs)
    queue   = Vector{Variable}()
    sorted  = Vector{Variable}()
    idegree = indegree(topnode; by=by)

    push!(queue, topnode)
    while length(queue) ≠ 0
        node = popfirst!(queue)
        if haskid(node)
            for kid in kidsof(node)
                idegree[kid] -= 1
                if idegree[kid] == 0
                    push!(queue, kid)
                end
            end
        end
        push!(sorted, node)
    end
    return sorted
end
