export AsyncLoader


"""
    AsyncLoader(dloader::DataLoader)

The `AsyncLoader` pulls minibatch of data from the `DataLoader`
and it accelerates training by loading minibatch asynchronously.
`AsyncLoader` can be indexed or iterated for training just the
same way as `DataLoader`. Julia is better started with more than
one thread.
"""
mutable struct AsyncLoader
    dataloader::DataLoader
    imax :: Int
    this :: Any
    next :: Any
    function AsyncLoader(dloader::DataLoader)
        new(dloader, length(dloader), nothing, dloader[1])
    end
end


Base.length(loader::AsyncLoader)     = loader.imax
Base.lastindex(loader::AsyncLoader)  = loader.imax
Base.firstindex(loader::AsyncLoader) = 1


function Base.getindex(loader::AsyncLoader, i::Int)
    # point to the already prepared data in the last round
    loader.this = loader.next
    # at the end, `next` would point to the first data agian
    if i == loader.imax
        @async loader.next = loader[1]
        return loader.this
    end
    # `next` should point to the (i+1)-th data
    @async loader.next = loader[i+1]
    return loader.this
end



function Base.iterate(loader::AsyncLoader, i::Int=1)
    i > loader.imax && return nothing
    return (loader[i], i+1)
end
