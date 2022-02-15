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
    sort_by_indegree(topnode::Variable) -> stack
"""
function sort_by_indegree(topnode::Variable; by::Function=indegree_by_bfs)
    sorted  = Vector{Variable}()
    record  = Vector{Variable}()
    idegree = indegree(topnode; by=by)

    while length(idegree) ≠ 0
        for (node, ins) in idegree
            if ins == 0
                push!(record, node)
                push!(sorted, node)
                if haskid(node)
                    for kid in kidsof(node)
                        idegree[kid] -= 1
                    end
                end
            end
        end
        # remove nodes with 0 indegree
        while length(record) ≠ 0
            delete!(idegree, pop!(record))
        end
    end
    return sorted
end
