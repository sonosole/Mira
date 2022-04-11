export @index
export @basic
export @extend

macro index(xstruct, blocks)
    quote
        NerualStruct = $(esc(xstruct))
        Base.length(s::NerualStruct)     = length(s.$blocks)
        Base.lastindex(s::NerualStruct)  = length(s.$blocks)
        Base.firstindex(s::NerualStruct) = 1
        Base.getindex(s::NerualStruct, k...)     =  s.$blocks[k...]
        Base.setindex!(s::NerualStruct, v, k...) = (s.$blocks[k...] = v)
        Base.iterate(s::NerualStruct, i=firstindex(s)) = i>length(s) ? nothing : (s[i], i+1)
    end
end

macro basic(xstruct, blocks)
    quote
        NerualStruct = $(esc(xstruct))

        function Base.show(io::IO, s::NerualStruct)
            print(io, "$NerualStruct(\n")
            join(io, s.$blocks, "\n")
            print(io, "\n)")
        end

        function Mira.paramsof(m::NerualStruct)
            params = Vector{Variable}(undef,0)
            for i = 1:length(m)
                append!(params, paramsof(m[i]))
            end
            return params
        end

        function Mira.xparamsof(m::NerualStruct)
            xparams = Vector{XVariable}(undef,0)
            for i = 1:length(m)
                append!(xparams, xparamsof(m[i]))
            end
            return xparams
        end

        function Mira.nparamsof(m::NerualStruct)
            nparams = 0
            for i = 1:length(m)
                nparams += nparamsof(m[i])
            end
            return nparams
        end

        Mira.elsizeof(m::NerualStruct) = elsizeof(m[1])

        function Mira.bytesof(model::NerualStruct, unit::String="MB")
            n = nparamsof(model) * elsizeof(model)
            return blocksize(n, uppercase(unit))
        end
        
        Mira.kbytesof(model::NerualStruct) = Mira.bytesof(model, "KB")
        Mira.mbytesof(model::NerualStruct) = Mira.bytesof(model, "MB")
        Mira.gbytesof(model::NerualStruct) = Mira.bytesof(model, "GB")
        Mira.tbytesof(model::NerualStruct) = Mira.bytesof(model, "TB")

        function Mira.nops(model::NerualStruct)
            mops, aops, acts = 0, 0, 0
            for m in model
                mo, ao, ac = nops(m)
                mops += mo
                aops += ao
                acts += ac
            end
            return (mops, aops, acts)
        end
    end
end

macro extend(xstruct, blocks)
    quote
        NerualStruct = $(esc(xstruct))
        Base.length(s::NerualStruct)     = length(s.$blocks)
        Base.lastindex(s::NerualStruct)  = length(s.$blocks)
        Base.firstindex(s::NerualStruct) = 1
        Base.getindex(s::NerualStruct, k...)     =  s.$blocks[k...]
        Base.setindex!(s::NerualStruct, v, k...) = (s.$blocks[k...] = v)
        Base.iterate(s::NerualStruct, i=firstindex(s)) = i>length(s) ? nothing : (s[i], i+1)

        function Base.show(io::IO, s::NerualStruct)
            print(io, "$NerualStruct(\n")
            join(io, s.$blocks, "\n")
            print(io, "\n)")
        end

        function Mira.paramsof(m::NerualStruct)
            params = Vector{Variable}(undef,0)
            for i = 1:length(m)
                append!(params, paramsof(m[i]))
            end
            return params
        end

        function Mira.xparamsof(m::NerualStruct)
            xparams = Vector{XVariable}(undef,0)
            for i = 1:length(m)
                append!(xparams, xparamsof(m[i]))
            end
            return xparams
        end

        function Mira.nparamsof(m::NerualStruct)
            nparams = 0
            for i = 1:length(m)
                nparams += nparamsof(m[i])
            end
            return nparams
        end

        Mira.elsizeof(m::NerualStruct) = elsizeof(m[1])

        function Mira.bytesof(model::NerualStruct, unit::String="MB")
            n = nparamsof(model) * elsizeof(model)
            return blocksize(n, uppercase(unit))
        end

        function Mira.nops(model::NerualStruct)
            mops, aops, acts = 0, 0, 0
            for m in model
                mo, ao, ac = nops(m)
                mops += mo
                aops += ao
                acts += ac
            end
            return (mops, aops, acts)
        end
    end
end
