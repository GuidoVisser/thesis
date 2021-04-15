#!/bin/bash

#SBATCH -n 1
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=gtx1080ti:4

# load modules
module load 2020
module load Python

# instasll dependencies
pip install tqdm
pip install --upgrade tensorboard && pip install --upgrade torch

#Copy input file to scratch
cp -RT $HOME/code/datasets/DAVIS/JPEGImages/480p "$TMPDIR/data"
cp -RT $HOME/code/datasets/DAVIS/Annotation/480p "$TMPDIR/masks"
cp -RT $HOME/code/FGVC/weight/zip_serialization_false/ "$TMPDIR/weight"

#Create output directory on scratch
mkdir "$TMPDIR"/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
echo "Start: $(date)" >> $HOME/job_logs/vae_train.log
python $HOME/code/VAE_mask_propagation.py \
            --data_dir $TMPDIR/data \
            --mask_dir $TMPDIR/masks \
            --log_dir $TMPDIR/output_dir \
            --flow_model $TMPDIR/weight/raft-things.pth \
            --epochs 10
echo "End: $(date)" >> $HOME/job_logs/vae_train.log

#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME/code/results/
