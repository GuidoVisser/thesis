#!/bin/bash

#SBATCH -n 1
#SBATCH -t 3:00:00
#SBATCH -p gpu_short
#SBATCH --gpus-per-node=gtx1080ti:1

# load modules
module load 2020
module load Python

#Copy input file to scratch
cp -RT $HOME/code/datasets/DAVIS/ $TMPDIR/data
cp -RT $HOME/code/FGVC/weight/zip_serialization_false/ $TMPDIR/weight

#Create output directory on scratch
mkdir $TMPDIR/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
echo "Start: $(date)" >> $HOME/job_logs/vae_train.log
python $HOME/code/DAVIS_flow.py \
        --data_root $TMPDIR/data \
        --flow_model $TMPDIR/weight
echo "End: $(date)" >> $HOME/job_logs/vae_train.log

#Copy output directory from scratch to home
cp -r $TMPDIR/data/Flow $HOME/code/datasets/DAVIS