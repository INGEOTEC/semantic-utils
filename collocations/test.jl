using TextSearch, SimilaritySearch, JSON, JLD2, DataFrames
using Languages, LIBLINEAR, KNearestCenters, Embeddings
using MLUtils, StatsBase
using UnicodePlots
import StatsAPI: predict, fit


include("io.jl")
include("vocab.jl")


struct BagOfSemanticWords{VectorModel,CLS}
    model::VectorModel
    cls::CLS
end

vectormodel(gw::EntropyWeighting, lw, corpus, labels, V) = VectorModel(gw, lw, V, corpus, labels)
vectormodel(gw, lw, corpus, labels, V) = VectorModel(gw, lw, V)


function fit(::Type{BagOfSemanticWords}, corpus, labels, tt=IdentityTokenTransformation();
        gw=EntropyWeighting(),
        lw=BinaryLocalWeighting(),
        collocations::Integer=0,
        nlist::Vector=[1],
        textconfig=TextConfig(; nlist, del_punc=false, del_diac=true, lc=true),
        qlist::Vector=[2, 3],
        mindocs::Integer=3,
        maxndocs::AbstractFloat=0.5
    )

    V = vocab(corpus, tt; collocations, nlist, qlist, mindocs, maxndocs)
    model = vectormodel(gw, lw, corpus, labels, V)
    X, y, dim = vectorize_corpus(model, corpus), labels, vocsize(model)
    BagOfSemanticWords(model, linear_train(y, sparse(X, dim)))
end

function predict_corpus(B::BagOfSemanticWords, corpus)
    Xtest = vectorize_corpus(B.model, corpus)
    dim = vocsize(B.model)
    pred, decision_value = linear_predict(B.cls, sparse(Xtest, dim))
    (; pred, decision_value)
end

function test(config) 
    train = read_json_dataframe(config.trainfile)
    test = read_json_dataframe(config.testfile)
    tt = Synonyms(config.mapfile)
    C = fit(BagOfSemanticWords, train.text, train.klass, tt; config.collocations, config.mindocs, config.qlist, config.gw, config.lw)
    y = predict_corpus(C, test.text)
    scores = classification_scores(test.klass, y.pred)
    @info json(scores, 2)
    
    (; config, scores,
       size=(voc=vocsize(C.model), train=size(train, 1), test=length(y.pred)),
       dist=(train=countmap(train.klass), test=countmap(test.klass), pred=countmap(y.pred))
    )
end

function embedding_by_lang(lang)
    if lang == "es"
        "embeddings/glove-embeddings-es-300d.h5"
    else
        error("I don't have support for lang $lang")
    end
end

function create_vocabmap(config; embfile=embedding_by_lang(config.lang), nick=config.nick, quantlist=[0.01, 0.03, 0.1, 0.3, 1])
    train = read_json_dataframe(config.trainfile)
    emb = load_emb(embfile)

    for mindocs in [1], quant in quantlist
        main_vocabmap(; outfile="map-$nick-$quant-$mindocs.json", quant, mindocs, train, emb)
    end
end

haha2018(; gw=EntropyWeighting(), lw=BinaryLocalWeighting(), collocations=7, mindocs=3, maxndocs=0.5, qlist=[2,5]) = (;
    trainfile = "datasets/competitions/haha2018_Es_train.json",
    testfile = "datasets/competitions/haha2018_Es_test.json",
    mapfile = "map-haha2018-0.01-1.json",
    nick = "haha2018",
    lang = "es",
    gw,
    lw,
    collocations,
    mindocs,
    maxndocs,
    qlist
   )

exist2021(; gw=IdfWeighting(), lw=BinaryLocalWeighting(), collocations=4, mindocs=2, maxndocs=0.5, qlist=[2,5]) = (;
    trainfile = "datasets/datasets/exist2021_task1_Es_train.json",
    testfile = "datasets/datasets/exist2021_task1_Es_test.json", 
    mapfile = "map-exist2021-1.0-1.json",
    nick = "exist2021",
    lang = "es",
    gw,
    lw,
    collocations,
    mindocs,
    maxndocs,
    qlist
   )

delitos(; gw=EntropyWeighting(), lw=BinaryLocalWeighting(), collocations=7, mindocs=3, maxndocs=0.5, qlist=[2,5]) = (;
    trainfile = "datasets/datasets/delitos_ingeotec_Es_train.json",
    testfile = "datasets/datasets/delitos_ingeotec_Es_test.json",
    mapfile = "map-delitos-0.01-1.json",
    nick = "delitos",
    lang = "es",
    gw,
    lw,
    collocations,
    mindocs,
    maxndocs,
    qlist
   )

