import EvoMSA
import json
import h5py

E = EvoMSA.BoW(voc_size_exponent=15)

with h5py.File("evomsa-bow-es-voc=15.h5", "w") as f:
    f.create_dataset("weights", data=E.weights)
    f.create_dataset("names", data=json.dumps(E.names))

