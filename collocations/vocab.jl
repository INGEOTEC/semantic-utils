include("../metric.jl")
using TextSearch, DataFrames

struct Synonyms <: AbstractTokenTransformation
    map::Dict{String,String}
    
    Synonyms(mapfile) = new(open(JSON.parse, mapfile))
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
    
    push!(sw, "_url")
    IgnoreStopwords(sw)
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

function vocab(
            text,
            tt=IdentityTokenTransformation();
            nlist=[1], qlist=[], collocations=0, mindocs=3, maxndocs=1.0, 
            textconfig=TextConfig(; nlist, del_punc=false, del_diac=true, lc=true)
    )
    #=tt = if T === IgnoreStopwords
        sw = spanish_stopwords(tc)
        IgnoreStopwords(sw)
    elseif T === Synonyms
        Synonyms(open(JSON.parse, mapfile))
    elseif T === ChainTransformation
        sw = spanish_stopwords(tc)
        ChainTransformation([IgnoreStopwords(sw), Synonyms(open(JSON.parse, mapfile))])
    else
        IdentityTokenTransformation()
    end=#
    
    V = Vocabulary(TextConfig(textconfig; qlist, collocations, tt), text)

    filter_tokens(V) do t
        mindocs <= t.ndocs < trainsize(V) * maxndocs
    end
end

function simvocab(names, vocab, embeddings, dist; verbose=true, k=12, minrecall=0.95)
    P = Dict(w => i for (i, w) in enumerate(vocab))
    ivocabidx = Int32[]
    for w in names
        i = get(P, w, 0)
        i > 0 && push!(ivocabidx, i)
    end

    X = StrideMatrixDatabase(embeddings)
    Y = StrideMatrixDatabase(embeddings[:, ivocabidx])
    G = create_index(dist, Y; k, minrecall, verbose)
    
    n = length(vocab)
    knns, dists = searchbatch(G, X, k)
    ivocab = vocab[ivocabidx]

    (; G, vocab, ivocab, ivocabidx, knns, dists)
end

function distquantiles(dist::SemiMetric, X::AbstractDatabase, q=[0.0, 0.25, 0.5, 0.75, 1.0]; samplesize=2^20)
    n = length(X)
    S = Vector{Float32}(undef, samplesize)
    
    Threads.@threads for i in 1:samplesize
        S[i] = evaluate(dist, X[rand(1:n)], X[rand(1:n)])
    end

    quantile(S, q)
end

function mapvocab(p::NamedTuple, dmax)
    println("creating map")
    map = Dict{String,String}()
    
    for i in 1:size(p.knns, 2)
       nn = p.knns[1, i]
       dist = p.dists[1, i]
       if dist <= dmax
           map[p.vocab[i]] = p.ivocab[nn]
       end
    end

    map

end


function main_vocabmap(; outfile::String, quant::AbstractFloat, emb::NamedTuple, train::DataFrame, mindocs::Integer, maxndocs=0.5)
    # train = read_json_dataframe(datafile) 
    # emb = load_fasttext_embeddings(embeddingsfile)
    voc = vocab(train.text; nlist=[1], qlist=[], collocations=0, mindocs, maxndocs)
    # @show voc, vocsize(voc)
    s = simvocab(token(voc), emb.vocab, emb.embeddings, emb.dist)
    dmax = distquantiles(emb.dist, StrideMatrixDatabase(emb.embeddings), quant)
    map = mapvocab(s, dmax)

    open(outfile, "w") do f
        print(f, json(map, 2))
    end

    map
end


