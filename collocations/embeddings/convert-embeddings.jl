using Embeddings, JLD2, LinearAlgebra


function convert_embeddings(embfile; output)
    E = load_embeddings(FastText_Text, embfile) 
    # E.embeddings, E.vocab
    for c in eachcol(E.embeddings)
        normalize!(c)
    end

    jldsave(output; E.vocab, E.embeddings)
end
