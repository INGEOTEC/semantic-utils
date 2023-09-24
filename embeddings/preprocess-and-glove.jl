using TextSearch, CodecZlib, JSON, Glowe

function preprocess_lines_and_save(text::Function, f::IO, tc::TextConfig, lines, mintokens=10)
    Threads.@threads for i in 1:length(lines)
        t = tokenize(tc, text(lines[i]))
        lines[i] = length(t.tokens) < mintokens ? nothing : join(t.tokens, ' ')
    end 

    @info "saving parsed lines"
    for line in lines
        line !== nothing && println(f, line)
    end
end

basictextconfig() = TextConfig(; nlist=[1], del_diac=true, lc=true, del_punc=false, group_usr=true)

function preprocess_large_compressed_json(corpusfile; prob=0.2, bsize=2^18,
                    output="corpus-preprocessed-from-json.txt",
                    tc=basictextconfig())
    open(output, "w") do f
        open(GzipDecompressorStream, corpusfile) do stream
            lines = []
            advance = 0
            for line in eachline(stream)
                rand() < prob && push!(lines, line)
                if bsize == length(lines)
                    advance += bsize
                    @info "**** preprocessing lines, advance: $advance"
                    preprocess_lines_and_save(f, tc, lines) do line_
                        JSON.parse(line_)["text"]
                    end

                    empty!(lines)
                end
            end

            if length(lines) > 0
                preprocess_lines_and_save(f, tc, lines) do line_
                    JSON.parse(line_)["text"]
                end
            end
        end
    end
end

function preprocess(files::AbstractVector;
                    output="corpus-preprocessed.txt",
                    tc=basictextconfig())
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

#=function vocab()
    tc = TextConfig(; nlist=[1], lc=false, del_diac=false) 
	  Vocabulary(tc, eachline("wit-es.txt"))
end=#

#using Glowe # please install it before running this function
function main_glove(corpusfile; output, min_count=5, memory=64.0, max_vocab=1_000_000, dim=300)
    vocab_count(corpusfile, "vocab.txt"; min_count, max_vocab, verbose=1)
    cooccur(corpusfile, "vocab.txt", "cooccurrence.bin"; memory, verbose=1)
    shuffle("cooccurrence.bin", "cooccurrence.shuf.bin"; memory, verbose=1)
    glove("cooccurrence.shuf.bin", "vocab.txt", output; threads=32,
          x_max=10.0, iter=15, vector_size=dim, binary=0, write_header=1, verbose=1)
end
