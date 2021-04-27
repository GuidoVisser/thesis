#!/bin/bash

#SBATCH -n 1
#SBATCH -t 3:00:00
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=gtx1080ti:1

# load modules
module load 2020
module load Python

# instasll dependencies
pip install --upgrade tensorboard && pip install --upgrade torch

#Copy input file to scratch
cp -RT $HOME/thesis/datasets/DAVIS_sample_tennis $TMPDIR/data
cp -RT $HOME/thesis/models/weights/zip_serialization_false/ $TMPDIR/weights

#Create output directory on scratch
mkdir $TMPDIR/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
echo "Start: $(date)" >> $HOME/job_logs/vae_train.log
python $HOME/thesis/train_sequential_VAE_mask_propagation.py \
            --data_dir $TMPDIR/data \
            --log_dir $TMPDIR/output_dir \
            --RAFT_weights $TMPDIR/weights
            --epochs 10
echo "End: $(date)" >> $HOME/thesis/job_logs/SeqMaskPropVAE.log

#Copy output directory from scratch to home
cp -RT $TMPDIR/output_dir $HOME/thesis/results/SeqMaskPropVAE
