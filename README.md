# A set of utitilies to work with semantic representations

##  Setup python environment

```bash
conda create -n "semantic"
conda install -n semantic -c conda-forge sentence-transformers h5py

```

## Julia setup
- Install julia
- Initialize environment
```bash

JULIA_PROJECT=. JULIA_NUM_THREADS=auto julia -e 'using Pkg; Pkg.initialize()'
```

run
```bash

JULIA_PROJECT=. JULIA_NUM_THREADS=auto julia -L metric.jl 
```

you should call the desired functions inside the Julia REPL


# About the SBERT

The encoder uses [SBERT](https://www.sbert.net/examples/training/multilingual/README.html) models.
