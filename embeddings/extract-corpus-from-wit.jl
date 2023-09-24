using Glob, DataFrames, CSV, TidierData

function main(; lang)
    text = []
    for filename in glob("WIT/ex_wit_v1.train.all-0000*.tsv")
        DATA = CSV.read(filename, DataFrame)
        @info filename
        en = @chain DATA begin
            @filter(language == lang)
            @select(context_page_description)
        end

        @info size(DATA), size(en)
        append!(text, en.context_page_description)
    end
        
    open("wit-$lang.txt", "w") do f
        for line in text
            println(f, line)
        end
    end
end
