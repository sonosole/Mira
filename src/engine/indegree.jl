export indegree
export resetindegree


function indegree(entry::Variable; by::Function=visit_by_dfs)
    # traverse dag
    nodes = by(entry)
    # first zero init
    idegree = Dict{Variable,Int}()
    for node in nodes
        push!(idegree, node => 0)
    end
    # then calculate indegree
    for (node, ins) in idegree
        if haskid(node)
            for kid in kidsof(node)
                idegree[kid] += 1
            end
        end
    end
    return idegree
end


function resetindegree(entry::Variable; by::Function=visit_by_dfs)
    # traverse dag
    nodes = by(entry)
    # first zero init
    for node in nodes
        node.indegree = 0
    end
    # then calculate indegree
    for node in nodes
        if haskid(node)
            for kid in kidsof(node)
                kid.indegree += 1
            end
        end
    end
end
