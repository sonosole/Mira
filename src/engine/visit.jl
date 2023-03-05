export visit_by_bfs
export visit_by_dfs


function visit_by_bfs(entry::Variable)
    store = Vector{Variable}()
    queue = Vector{Variable}()

    push!(queue, entry)
    setmarked(entry)
    while notempty(queue)
        node = popfirst!(queue)
        push!(store, node)
        if haskid(node)
            for kid in kidsof(node)
                if !ismarked(kid)
                    push!(queue, kid)  # once pushed,
                    setmarked(kid)     # set marked.
                end
            end
        end
    end

    for node in store
        unsetmarked(node)
    end
    return store
end


function visit_by_dfs(entry::Variable)
    store = Vector{Variable}()
    stack = Vector{Variable}()

    push!(stack, entry)
    setmarked(entry)
    while notempty(stack)
        node = pop!(stack)
        push!(store, node)
        if haskid(node)
            for kid in kidsof(node)
                if !ismarked(kid)
                    push!(stack, kid)  # once pushed,
                    setmarked(kid)     # set marked.
                end
            end
        end
    end

    for node in store
        unsetmarked(node)
    end
    return store
end
