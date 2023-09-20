import fasttext
import sys
import os

filename = sys.argv[1]
dim = 300
outname = filename.replace(".txt", "") + "-{}d.bin".format(dim)

if not os.path.isfile(outname):
    model = fasttext.train_unsupervised(filename, model="skipgram", dim=dim)
    model.save_model(outname)
