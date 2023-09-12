
function read_json_dataframe(filename)
    L = open(filename) do f
        [JSON.parse(line) for line in eachline(f)]
    end

    DataFrame(L)
end

function read_synonyms(filename)
    Dict(k => first(v) for (k, v) in open(JSON.parse, filename))
    #syn = Dict(k => Vector(v) for (k, v) in open(JSON.parse, "map-fastText-to-haha.json"))
end

