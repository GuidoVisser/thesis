#!/bin/bash

#SBATCH -n 1
#SBATCH -t 15:00:00
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=gtx1080ti:1

# load modules
module load 2020
module load Python

# instasll dependencies
pip install --user --upgrade tensorboard && pip install --user --upgrade torch && pip install --user --upgrade torchvision
pip install moviepy

#Copy input file to scratch
cp -RT $HOME/thesis/datasets/DAVIS $TMPDIR/data

#Create output directory on scratch
mkdir $TMPDIR/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
echo "Start: $(date)" >> $HOME/thesis/job_logs/RecMaskPropVAE.log
python $HOME/thesis/train_recurrent_VAE_mask_propagation.py \
            --data_dir $TMPDIR/data \
            --log_dir $TMPDIR/output_dir \
            --epochs 50 \
            --batch_size 1 \
            --seq_length 8
echo "End: $(date)" >> $HOME/thesis/job_logs/RecMaskPropVAE.log

#Copy output directory from scratch to home
cp -RT $TMPDIR/output_dir $HOME/thesis/results/RecMaskPropVAE
