struct Synonyms <: AbstractTokenTransformation
    # map::Dict{String,Vector{String}}
    map::Dict{String,String}
end

function TextSearch.transform_unigram(tt::Synonyms, tok)
    get(tt.map, tok, tok)
end

function spanish_stopwords(textconfig)
    sw = Set{String}()
    sp = Languages.Spanish()
    for w in stopwords(sp)
        push!(sw, tokenize(first, textconfig, w))
    end
    for w in articles(sp)
        push!(sw, tokenize(first, textconfig, w))
    end
    for w in prepositions(sp)
        push!(sw, tokenize(first, textconfig, w))
    end
    for w in pronouns(sp)
        push!(sw, tokenize(first, textconfig, w))
    end
    sw
end

struct IgnoreStopwords <: AbstractTokenTransformation
    stopwords::Set{String}
end

function TextSearch.transform_unigram(tt::IgnoreStopwords, tok)
    tok in tt.stopwords ? nothing : tok
end

struct ChainTransformation <: AbstractTokenTransformation
    list::AbstractVector{<:AbstractTokenTransformation}
end

function TextSearch.transform_unigram(tt::ChainTransformation, tok)
    for t in tt.list
        tok = TextSearch.transform_unigram(t, tok)
        tok === nothing && return nothing
    end 

    tok
end
