#!/bin/bash

#SBATCH --job-name=a2p_var                     # sets the job name
#SBATCH --output=a2p_var.%j                    # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=a2p_var.%j                     # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=08:00:00                            # how long you would like your job to run; format=hh:mm:ss

#SBATCH --partition=scavenger
#SBATCH --qos=scavenger                            # set QOS, this will determine what resources can be requested
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa5000:1

#SBATCH --nodes=1                                  # number of nodes to allocate for your job
#SBATCH --ntasks=1                                             
#SBATCH --ntasks-per-node=1                                     
#SBATCH --mem=64gb                                 # (cpu) memory required by job; if unit is not specified MB will be assumed

module load cuda
module load ffmpeg
source ~/.bashrc
micromamba activate a2p_env

python inference.py --person_id PXB184 \
                    --input_audio audio/talkshow_various_char_trimmed.wav \
                    --output_dir outputs/talkshow_var/PXB184 \
                    --num_samples 3

python inference.py --person_id GQS883 \
                    --input_audio audio/talkshow_various_char_trimmed.wav \
                    --output_dir outputs/talkshow_var/GQS883 \
                    --num_samples 3

python inference.py --person_id RLW104 \
                    --input_audio audio/talkshow_various_char_trimmed.wav \
                    --output_dir outputs/talkshow_var/RLW104 \
                    --num_samples 3

python inference.py --person_id TXB805 \
                    --input_audio audio/talkshow_various_char_trimmed.wav \
                    --output_dir outputs/talkshow_var/TXB805 \
                    --num_samples 3

                    