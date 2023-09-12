using Embeddings

include("../metric.jl")

function load_fasttext_embeddings(embfile)
    E = load_embeddings(FastText_Text, embfile) 
    # E.embeddings, E.vocab
    dist = NormalizedCosineDistance()
    for c in eachcol(E.embeddings)
        normalize!(c)
    end

    (; E.vocab, E.embeddings, dist)
end


function similarity_synonyms(names, vocab, embeddings, dist; verbose=true, k=16, minrecall=0.95)
    P = Dict(w => i for (i, w) in enumerate(vocab))
    ovocab = String[]
    ivocab = String[]
    ivocabidx = Int32[]
    for w in names
        i = get(P, w, 0)
        if i > 0
            push!(ivocab, w)
            push!(ivocabidx, i)
        else
            push!(ovocab, w)
        end
    end

    X = StrideMatrixDatabase(embeddings)
    Y = StrideMatrixDatabase(embeddings[:, ivocabidx])
    G = create_index(dist, Y; k, minrecall, verbose)
    
    n = length(vocab)
    R = Vector{Int32}(undef, n)
    knns, dists = searchbatch(G, X, k)

    (; G, vocab, ivocab, ovocab, knns, dists)
end

function create_map(p::NamedTuple, k::Integer=4)
    println("creating map")
    map = Dict{String,Vector{String}}()
    
    for i in 1:size(p.knns, 2)
       I = @view p.knns[:, i]
       map[p.vocab[i]] = p.ivocab[I[1:k]]
    end

    for w in p.ovocab
        map[w] = [String(w)]
    end

    map

    #=
    open("map-fastText-to-evomsa-17.json", "w") do f
           print(f, json(map, 2))
    end
    =#
end

