using Embeddings, Graphs

include("../metric.jl")

function load_fasttext_embeddings(embfile)
    E = load_embeddings(FastText_Text, embfile) 
    # E.embeddings, E.vocab
    dist = NormalizedCosineDistance()
    for c in eachcol(E.embeddings)
        normalize!(c)
    end

    E.vocab, E.embeddings, dist
end


function fastText_knngraph(embfile)
    vocab, embeddings, dist = load_fasttext_embeddings(embfile)
    k = 16
    X = StrideMatrixDatabase(embeddings)
    G = create_index(dist, X; k, minrecall=0.9, verbose=true)
    vocab, allknn(G, k)...
end

function create_graph(knns)
    G = Graph{Int32}()
    add_vertices!(G, size(knns, 2))
    for (i, neighbors) in enumerate(eachcol(knns))
        for j in neighbors
            i != j && add_edge!(G, i, j)
        end
    end

    G
end

function shortest_path_synonymous()
    names, weights = jldopen("evomsa/evomsa-bow-es-voc=17.h5") do f
        JSON.parse(f["names"]), f["weights"]
    end

    vocab, knns, dists = load("ALL-knngraph.jld2", "vocab", "knns", "dists")
    P = Dict(w => i for (i, w) in enumerate(vocab))
    ovocab = String[]
    ivocab = String[]
    for w in names
        if haskey(P, w)
            push!(ivocab, w)
        else
            push!(ovocab, w)
        end
    end

    G = create_graph(knns)
    regions = [KnnResult(1) for _ in 1:length(vocab)]

    DD = Vector(undef, length(ivocab))
    lock_ = Threads.SpinLock()
    counter = 0
    for part in Iterators.partition(1:length(ivocab), 128)
        GC.gc()
        Threads.@threads for i in part
            w = ivocab[i]
            ## D = dijkstra_shortest_paths(G, P[w])
            DD[i] = dijkstra_shortest_paths(G, P[w])
            ## D = dijkstra_shortest_paths(G, P[w])
            
            try
                lock(lock_)
                counter += 1
                println("dijkstra_shortest_paths i: $i, word: $w -- advance $counter of $(length(ivocab))")
            finally
                unlock(lock_)
            end
        end

        for i in part
            D = DD[i]
            for (j, d) in enumerate(D.dists)
                push_item!(regions[j], i, d)
            end
            # i == 100 && break
        end
    end

    (; G, vocab, ivocab, ovocab, regions)
end

function similarity_synonymous()
    names, weights = jldopen("evomsa/evomsa-bow-es-voc=17.h5") do f
        JSON.parse(f["names"]), f["weights"]
    end

    vocab, knns, dists = load("ALL-knngraph.jld2", "vocab", "knns", "dists")
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

    vocab, embeddings, dist = load_fasttext_embeddings("regional-spanish-models/ALL.vec")
    k = 16
    X = StrideMatrixDatabase(embeddings)
    Y = StrideMatrixDatabase(embeddings[:, ivocabidx])
    G = create_index(dist, Y; k, minrecall=0.95, verbose=true)
    
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

