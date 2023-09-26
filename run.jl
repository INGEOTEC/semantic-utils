using BagOfWords, TextSearch, JLD2, DataFrames, CSV, MLUtils, StatsBase, KNearestCenters, Random

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

