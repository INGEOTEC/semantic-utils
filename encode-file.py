from sentence_transformers import SentenceTransformer
import h5py
import json
import gzip
import sys
import os
import argparse

print("** loading model")

modelname = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
#'sentence-transformers/all-MiniLM-L6-v2'
modelnick = "sbert-multi-L12-v2"
model = SentenceTransformer(modelname)

def json_tweets(filename, output=None, key="text"):
    T = []
    if output is None:
        output = os.path.basename(filename) + f"--{modelnick}.h5"

    print(f"** reading {filename} dataset")
    with open(filename) as f:
        for line in f:
            tweet = json.loads(line)
            text = tweet[key]
            # do some preprocessing?
            T.append(text)

    assert len(T) > 0, f"ERROR {filename} is empty"
    print(T[:10])

    print(f"** encoding with {modelname}")
    emb = model.encode(T)
    
    print(f"** saving embeddings in {output}")
    with h5py.File(output, "w") as f:
        f.attrs['filename'] = filename
        f.attrs['modelname'] = modelname 
        f.attrs['modelnick'] = modelnick
        f.create_dataset("emb", emb.shape, dtype=emb.dtype)[:] = emb
     
