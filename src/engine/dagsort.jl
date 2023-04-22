export sort_by_recursive_dfs
export sort_by_bfs
export sort_by_dfs
export level_sort_by_bfs
export level_sort_by_dfs


"""
    sort_by_recursive_dfs(entry::Variable) -> stack::Vector{Variable}
"""
function sort_by_recursive_dfs(entry::Variable)
    stack = Vector{Variable}()

    function visit(node::Variable)
        setmarked(node)
        if haskid(node)
            for kid in kidsof(node)
                if !ismarked(kid)
                    visit(kid)
                end
            end
        end
        push!(stack, node)
    end

    visit(entry)
    return stack
end



"""
    sort_by_bfs(entry::Variable) -> sorted::Vector{Variable}
"""
function sort_by_bfs(entry::Variable)
    @assert isroot(entry) "not a root node"
    sorted = Vector{Variable}()
    queue  = Vector{Variable}()

    push!(queue, entry)
    while notempty(queue)
        node = popfirst!(queue)
        push!(sorted, node)
        if haskid(node)
            for kid in kidsof(node)
                kid.indegree -= 1
                if kid.indegree == 0
                    push!(queue, kid)
                end
            end
        end
    end

    return sorted
end


"""
    sort_by_dfs(entry::Variable) -> sorted::Vector{Variable}
"""
function sort_by_dfs(entry::Variable)
    @assert isroot(entry) "not a root node"
    sorted = Vector{Variable}()
    stack  = Vector{Variable}()

    push!(stack, entry)
    while notempty(stack)
        node = pop!(stack)
        push!(sorted, node)
        if haskid(node)
            for kid in kidsof(node)
                kid.indegree -= 1
                if kid.indegree == 0
                    push!(stack, kid)
                end
            end
        end
    end

    return sorted
end


"""
    level_sort_by_bfs(entry::Variable) -> sorted::Vector{Vector{Variable}}
"""
function level_sort_by_bfs(entry::Variable)
    @assert isroot(entry) "not a root node"
    sorted = Vector{Vector{Variable}}()
    queue  = Vector{Vector{Variable}}()

    push!(queue, Variable[entry])
    while !isempty(queue)
        level = Vector{Variable}()
        cells = popfirst!(queue)
        push!(sorted, cells)
        for cell in cells
            if haskid(cell)
                for kid in kidsof(cell)
                    kid.indegree -= 1
                    if kid.indegree == 0
                        push!(level, kid)
                    end
                end
            end
        end
        if !isempty(level)
            push!(queue, level)
        end
    end

    return sorted
end


"""
    level_sort_by_dfs(entry::Variable) -> sorted::Vector{Vector{Variable}}
"""
function level_sort_by_dfs(entry::Variable)
    @assert isroot(entry) "not a root node"
    sorted = Vector{Vector{Variable}}()
    stack  = Vector{Vector{Variable}}()

    push!(stack, Variable[entry])
    while !isempty(stack)
        level = Vector{Variable}()
        cells = pop!(stack)
        push!(sorted, cells)
        for cell in cells
            if haskid(cell)
                for kid in kidsof(cell)
                    kid.indegree -= 1
                    if kid.indegree == 0
                        push!(level, kid)
                    end
                end
            end
        end
        if !isempty(level)
            push!(stack, level)
        end
    end

    return sorted
end
