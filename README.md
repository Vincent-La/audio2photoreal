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

## `inference.py`
Perform end-to-end inference generating photorealistic avatar render from audio

Usage:
```
usage: inference.py [-h] --person_id {PXB184,RLW104,TXB805,GQS883} --input_audio INPUT_AUDIO --output_dir OUTPUT_DIR [--num_samples NUM_SAMPLES]
                    [--sample_diversity SAMPLE_DIVERSITY] [--guidance_param GUIDANCE_PARAM]

optional arguments:
  -h, --help            show this help message and exit
  --person_id {PXB184,RLW104,TXB805,GQS883}
                        Person ID to select face, pose, guide, and avatar models
  --input_audio INPUT_AUDIO
                        Path to audio (.wav) file as input
  --output_dir OUTPUT_DIR
                        Path to output files to
  --num_samples NUM_SAMPLES
                        Number of samples to generate
  --sample_diversity SAMPLE_DIVERSITY
                        Tunes the cumulative probability in nucleus sampling: 0.01 = low diversity, 1.0 = high diversity.
  --guidance_param GUIDANCE_PARAM
                        how influential the conditioning is on the results, reccommended [2.0, 10.0]
```

`--guidance_param` only changes the value for the body/guide models for now. Face model value is set to 10.0

See [submit_scripts](submit_scripts) for examples of using inference.py
