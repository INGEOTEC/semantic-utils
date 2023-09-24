using TextSearch, SimilaritySearch, JSON, JLD2, DataFrames
using Languages, LIBLINEAR, StatsBase, Printf, Glob
using UnicodePlots

include("test.jl")
include("synonyms.jl")


function join_vocabularies(files; mindocs, collocations=false)
    E = nothing
    for filename in files
        voc = load(filename, "voc")
        if !collocations
            voc = filter_tokens(voc) do t
                endswith(t.token, "\tc")
            end
        end

        if E === nothing 
            E = voc
        else
            update_voc!(E, voc)
        end
    end
    
    filter_tokens(E) do t
        mindocs <= t.ndocs
    end
end

function subembeddings()
    voc = load("voc/voc-ALL-10M.jld2", "voc")
    emb = load_fasttext_embeddings("regional-spanish-models/ALL-1.7m.vec")

    s = similarity_synonyms(voc.token, emb.vocab, emb.embeddings, emb.dist; k=8)
    map = create_map(s; quant=0.4)
    subemb = emb.embeddings[:, s.ivocabidx] |> MatrixDatabase
    word2id = Dict(v => k for (k, v) in enumerate(s.ivocab))
    W = Dict(w => word2id[map[w]] for (i, w) in enumerate(s.vocab) if haskey(map, w))
    # dot(subemb[W["rey"]], subemb[W["rey"]])
    (; emb=subemb, s.vocab, s.ivocab, s.ivocabidx, map)
end

function join_unigrams(files; mindocs, collocations=false)
    E = nothing
    for filename in files
        voc = load(filename, "voc")
        voc = filter_tokens(voc) do t
            endswith(t.token, "\tc")
        end

        if E === nothing 
            E = voc
        else
            update_voc!(E, voc)
        end
    end
    
    filter_tokens(E) do t
        mindocs <= t.ndocs
    end
end

function compute_vocabularies(
        filename;
        bsize = 10^6,
        mindocs = 5,
        maxndocs = 0.5,
        collocationsmindocs = 3,
        collocations = 7
    )

    L = eachline(filename)
    tc = TextConfig(; del_punc=true, nlist=[1], qlist=[], collocations=0)
    tt = IgnoreStopwords(spanish_stopwords(tc))
    tc = TextConfig(tc; collocations, tt) 
    i = 0
    while true
        i += 1
        @info "---- vocabulary $bsize - $(i)th iteration"
        corpus = Iterators.take(L, bsize) |> collect
        length(corpus) == 0 && break

        outfile = @sprintf "voc/voc-%s-%02d.jld2" replace(basename(filename), ".txt" => "") i
        if isfile(outfile)
            @info "skipping $outfile"
            continue
        end

        V = Vocabulary(tc, corpus)
        voc = filter_tokens(V) do t
            if endswith(t.token, "\tc")
                collocationsmindocs <= t.ndocs
            else
                mindocs <= t.ndocs <= length(V) * maxndocs
            end
        end

        jldsave(outfile; voc)
    end
end
