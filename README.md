# CMSC848M Vision and Speech Experiments

+ Original [README](original_README.md)

## Installation
Had some issues with `pytorch3d` and `fairseq` not interacting well with `pip` but this SHOULD work:

```
conda create --name a2p_env python=3.9 pip=24.0
conda activate a2p_env
pip install -r scripts/requirements.txt
sh scripts/download_prereq.sh

# Force the GPU version to be installed
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## `person_id` to model checkpoint names
These are named differently for some reason but the mapping is stored in some `.json` files as follows:

+ `PXB184` <--> `c1`
+ `RLW104` <--> `c2`
+ `TXB805` <--> `c3`
+ `GQS883` <--> `c4`
