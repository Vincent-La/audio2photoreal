#!/bin/bash

#SBATCH --job-name=a2p_visualize                     # sets the job name
#SBATCH --output=a2p_visualize.%j                    # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=a2p_visualize.%j                     # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=04:00:00                            # how long you would like your job to run; format=hh:mm:ss


#SBATCH --partition=scavenger
#SBATCH --qos=scavenger                            # set QOS, this will determine what resources can be requested
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa5000:1

#SBATCH --nodes=1                                  # number of nodes to allocate for your job
#SBATCH --ntasks=1                                             
#SBATCH --ntasks-per-node=1                                     
#SBATCH --mem=64gb                                 # (cpu) memory required by job; if unit is not specified MB will be assumed

module load cuda
source ~/.bashrc
micromamba activate a2p_env

python -m sample.generate \
       --model_path checkpoints/diffusion/c1_pose/model000340000.pt \
       --resume_trans checkpoints/guide/c1_pose/checkpoints/iter-0100000.pt \
       --num_samples 10 \
       --num_repetitions 5 \
       --timestep_respacing ddim500 \
       --guidance_param 2.0 \
       --face_codes ./checkpoints/diffusion/c1_face/samples_c1_face_000155000_seed10_/results.npy \
       --pose_codes ./checkpoints/diffusion/c1_pose/samples_c1_pose_000340000_seed10_guide_iter-0100000.pt/results.npy \
       --plot
