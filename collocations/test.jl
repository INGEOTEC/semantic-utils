using TextSearch, SimilaritySearch, JSON, JLD2, DataFrames
using Languages, LIBLINEAR, StatsBase, KNearestCenters, Embeddings
using UnicodePlots


include("io.jl")
include("voctrans.jl")

function vocab(::Type{T}, text; nlist, qlist, collocations, mindocs, maxndoc) where {T<:Union{AbstractTokenTransformation,Nothing}}
    tc = TextConfig(; nlist, del_diac=true, del_punc=true)
    tt = if T === IgnoreStopwords
        sw = spanish_stopwords(tc)
        IgnoreStopwords(sw)
    elseif T === Synonyms
        Synonyms(read_synonyms("map-fastText-to-haha.json"))
    elseif T === ChainTransformation
        sw = spanish_stopwords(tc)
        ChainTransformation([IgnoreStopwords(sw), Synonyms(read_synonyms("map-fastText-to-haha.json"))])
    else
        IdentityTokenTransformation()
    end
    
    V = Vocabulary(TextConfig(tc; qlist, collocations, tt), text)

    filter_tokens(V) do t
      mindocs < t.ndocs < trainsize(V) * maxndocs
    end
end

vectormodel(gw::EntropyWeighting, lw, train, V) = VectorModel(gw, lw, V, train.text, train.klass)
vectormodel(gw, lw, train, V) = VectorModel(gw, lw, V)

function textclassifier(train, test; gw=EntropyWeighting(), lw=BinaryLocalWeighting(), voctype=nothing, collocations=0, nlist=[1], qlist=[], mindocs=3, maxndocs=0.5)
    V = vocab(voctype, train.text; collocations, nlist, qlist, mindocs, maxndocs)
    model = vectormodel(gw, lw, train, V)
    X, y, dim = vectorize_corpus(model, train.text), train.klass, vocsize(model)
    cls = linear_train(y, sparse(X, dim))     
    Xtest = vectorize_corpus(model, test.text)
    ypred, ypred_decision = linear_predict(cls, sparse(Xtest, dim))
    @info json(classification_scores(test.klass, ypred), 2)
    model
end
 
train = read_json_dataframe("datasets/competitions/haha2018_Es_train.json")
test = read_json_dataframe("datasets/competitions/haha2018_Es_test.json")
# E = totable(model, DataFrame) 
