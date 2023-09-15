using Embeddings

include("../metric.jl")
include("io.jl")
include("test.jl")

function load_fasttext_embeddings(embfile)
    E = load_embeddings(FastText_Text, embfile) 
    # E.embeddings, E.vocab
    dist = NormalizedCosineDistance()
    for c in eachcol(E.embeddings)
        normalize!(c)
    end

    (; E.vocab, E.embeddings, dist)
end


function similarity_synonyms(names, vocab, embeddings, dist; verbose=true, k=12, minrecall=0.95)
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

function create_map(p::NamedTuple; quant=0.02)
    println("creating map")
    map = Dict{String,String}()
    dmax = quantile(p.dists[1, :], quant)
    
    for i in 1:size(p.knns, 2)
       nn = p.knns[1, i]
       dist = p.dists[1, i]
       if dist <= dmax
           map[p.vocab[i]] = p.ivocab[nn]
       end
       #I = @view p.knns[:, i]
       #D = @view p.dist[:, i]
       #map[p.vocab[i]] = p.ivocab[I[1:k]]
    end

    map

end


function main(; outfile::String, quant::AbstractFloat, emb::NamedTuple, train::DataFrame, mindocs::Integer, maxndocs=0.5)
    # train = read_json_dataframe(datafile) 
    # emb = load_fasttext_embeddings(embeddingsfile)
    voc = vocab(Nothing, train.text; nlist=[1], qlist=[], collocations=0, mindocs, maxndocs)
    dataset_tokens = token(voc)
    s = similarity_synonyms(dataset_tokens, emb.vocab, emb.embeddings, emb.dist)
    map = create_map(s; quant)
    
    open(outfile, "w") do f
           print(f, json(map, 2))
    end

    map
end
