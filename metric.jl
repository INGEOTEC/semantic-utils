using JSON, SimilaritySearch, JLD2, DataFrames, LinearAlgebra

function load_sentence_embeddings(filename; key, normalize)
    jldopen(filename) do f
        X = f[key]
        if normalize
            for c in eachcol(X)
                normalize!(c)
            end
        end

        X
    end
end

function knngraph(input::String; output::String, k::Int=16, key::String="emb", minrecall::Float64=0.95, normalize=true)
    X = load_sentence_embeddings(input; key, normalize)
    G = SearchGraph(dist=NormalizedCosineDistance(), db=MatrixDatabase(X))
    minrecall = MinRecall(minrecall)
    callbacks = SearchGraphCallbacks(minrecall)
    index!(G; callbacks)
    optimize!(G, minrecall)
    knns, dists = allknn(G, k)
    jldsave(output; knns, dists)
    knns, dists
end

    #=
        D = DataFrame(JSON.parse.(eachline("datasets/comp2023/IberLEF2023_HOMO-MEX_Es_train.json")))
D = DataFrame(JSON.parse.(eachline("datasets/comp2023/IberLEF2023_HOMO-MEX_Es_train.json")))

        for r in eachrow(D[knns[:, 1], :])
            println(r)
        end
# time: 2023-09-01 12:31:51 CST
# mode: julia
        for r in eachrow(D[knns[:, 1], :])
            println(r.klass => r.text)
        end
# time: 2023-09-01 12:36:19 CST
# mode: julia
        for r in eachrow(D[knns[:, 2], :])
            println(r.klass => r.text)
        end
        =#
