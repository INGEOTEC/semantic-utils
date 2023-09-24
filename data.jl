using TextSearch

function embedding_by_lang(lang)
    if lang == "es"
        "embeddings/glove-embeddings-es-300d.h5"
    elseif lang == "en"
        "embeddings/glove-embeddings-en-300d.h5"
    else
        error("I don't have support for lang $lang")
    end
end

haha2018(; gw=EntropyWeighting(), lw=BinaryLocalWeighting(), collocations=7, mindocs=3, maxndocs=0.5, qlist=[2,5], spelling=nothing) =
    (;
        trainfile = "datasets/competitions/haha2018_Es_train.json",
        testfile = "datasets/competitions/haha2018_Es_test.json",
        mapfile = "mapping/map-haha2018-0.01-1.json",
        nick = "haha2018",
        lang = "es",
        gw,
        lw,
        collocations,
        mindocs,
        maxndocs,
        qlist,
        spelling
   )

exist2021(; gw=IdfWeighting(), lw=BinaryLocalWeighting(), collocations=4, mindocs=2, maxndocs=0.5, qlist=[2,5], mapfile="mapping/map-exist2021-1.0-1.json") =
    (;
        trainfile = "datasets/datasets/exist2021_task1_Es_train.json",
        testfile = "datasets/datasets/exist2021_task1_Es_test.json",
        mapfile,
        nick = "exist2021",
        lang = "es",
        gw,
        lw,
        collocations,
        mindocs,
        maxndocs,
        qlist
   )

hope2023(; gw=EntropyWeighting(), lw=BinaryLocalWeighting(), collocations=10, mindocs=2, maxndocs=0.5, qlist=[4,5], mapfile=nothing) =
    (;
        trainfile = "datasets/comp2023/IberLEF2023_HOPE_Es_train.json",
        testfile = "datasets/comp2023/IberLEF2023_HOPE_Es_test.json",
        mapfile,
        nick = "hope2023",
        lang = "es",
        gw,
        lw,
        collocations,
        mindocs,
        maxndocs,
        qlist
   )

huhu2023(; gw=EntropyWeighting(), lw=BinaryLocalWeighting(), collocations=10, mindocs=3, maxndocs=0.5, qlist=[2,5], mapfile="mapping/map-huhu2023-0.01-1.json") =
    (;
        trainfile = "datasets/comp2023/IberLEF2023_HUHU_task1_Es_train.json",
        testfile = "datasets/comp2023/IberLEF2023_HUHU_task1_Es_test.json",
        mapfile,
        nick = "huhu2023",
        lang = "es",
        gw,
        lw,
        collocations,
        mindocs,
        maxndocs,
        qlist
   )


delitos(; gw=EntropyWeighting(), lw=BinaryLocalWeighting(), collocations=7, mindocs=3, maxndocs=0.5, qlist=[2,5], mapfile="mapping/map-delitos-0.01-1.json", spelling=nothing) =
    (;
        trainfile = "datasets/datasets/delitos_ingeotec_Es_train.json",
        testfile = "datasets/datasets/delitos_ingeotec_Es_test.json",
        mapfile,
        nick = "delitos",
        lang = "es",
        gw,
        lw,
        collocations,
        mindocs,
        maxndocs,
        qlist,
        spelling
   )

offenseval2019A(; gw=IdfWeighting(), lw=TfWeighting(), collocations=10, mindocs=2, maxndocs=0.5, qlist=[2,5], mapfile=nothing) =
    (;
        trainfile = "datasets/datasets/offenseval2019_A_En_train.json",
        testfile = "datasets/datasets/offenseval2019_A_En_test.json",
        mapfile,
        nick = "offenseval2019A",
        lang = "en",
        gw,
        lw,
        collocations,
        mindocs,
        maxndocs,
        qlist
   )

meoffendes2021task3(; gw=IdfWeighting(), lw=TfWeighting(), collocations=10, mindocs=2, maxndocs=0.5, qlist=[2,5], mapfile=nothing) =
    (;
        trainfile = "datasets/datasets/meoffendes2021_task3_Es_train.json",
        testfile = "datasets/datasets/meoffendes2021_task3_Es_test.json",
        mapfile,
        nick = "meoffendes2021task3",
        lang = "es",
        gw,
        lw,
        collocations,
        mindocs,
        maxndocs,
        qlist
   )

meoffendes2021task1(; gw=IdfWeighting(), lw=TfWeighting(), collocations=10, mindocs=2, maxndocs=0.5, qlist=[2,5], mapfile=nothing) =
    (;
        trainfile = "datasets/datasets/meoffendes2021_task1_Es_train.json",
        testfile = "datasets/datasets/meoffendes2021_task1_Es_test.json",
        mapfile,
        nick = "meoffendes2021task1",
        lang = "es",
        gw,
        lw,
        collocations,
        mindocs,
        maxndocs,
        qlist
   )

politicES2023_gender(; gw=EntropyWeighting(), lw=BinaryLocalWeighting(), collocations=10, mindocs=2, maxndocs=0.5, qlist=[2,5], mapfile=nothing) =
    (;
        trainfile = "datasets/comp2023/IberLEF2023_PoliticEs_gender_Es_train.json.gz",
        testfile = "datasets/comp2023/IberLEF2023_PoliticEs_gender_Es_test.json.gz",
        mapfile,
        nick = "PoliticES-gender",
        lang = "es",
        gw,
        lw,
        collocations,
        mindocs,
        maxndocs,
        qlist
   )
