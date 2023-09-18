using TextSearch

function preprocess(files::AbstractVector;
                    output="corpus-preprocessed.txt",
                    tc = TextConfig(; nlist=[1], del_diac=true, lc=true, del_punc=false))
                    #tc = TextConfig(; nlist=[1], del_diac=false, lc=false, del_punc=true))
    open(output, "w") do f
        for corpusfile in files
            @info "preparing $corpusfile"
            lines = readlines(corpusfile)
            Threads.@threads for i in 1:length(lines)
                t = tokenize(tc, lines[i])
                lines[i] = join(t.tokens, ' ')
            end 

            @info "saving parsed $corpusfile into $output"
            for line in lines
                println(f, line)
            end
        end
    end
end

function vocab()
    tc = TextConfig(; nlist=[1], lc=false, del_diac=false) 
	  Vocabulary(tc, eachline("wit-es.txt"))
end

#using Glowe # please install it before running this function
function main_glove(corpusfile; output, min_count=5, memory=32.0, max_vocab=500_000, dim=300)
    vocab_count(corpusfile, "vocab.txt"; min_count, max_vocab, verbose=1)
    cooccur(corpusfile, "vocab.txt", "cooccurrence.bin"; memory, verbose=1)
    shuffle("cooccurrence.bin", "cooccurrence.shuf.bin"; memory, verbose=1)
    glove("cooccurrence.shuf.bin", "vocab.txt", output; threads=32,
          x_max=10.0, iter=15, vector_size=dim, binary=0, write_header=1, verbose=1)
end
