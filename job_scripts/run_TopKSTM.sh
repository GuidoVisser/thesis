#!/bin/bash

#SBATCH -n 1
#SBATCH -t 00:05:00
#SBATCH -p gpu_short
#SBATCH --gpus-per-node=gtx1080ti:1

# load modules
module load 2020
module load Python

# instasll dependencies
pip install --user --upgrade tensorboard && pip install --user --upgrade torch && pip install --user --upgrade torchvision

#Copy input file to scratch
cp -RT $HOME/thesis/datasets/DAVIS_sample $TMPDIR/data
mkdir $TMPDIR/weights
cp $HOME/thesis/models/weights/MiVOS/propagation_model.pth $TMPDIR/weights/propagation_model.pth

#Create output directory on scratch
mkdir $TMPDIR/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
echo "Start: $(date)" >> $HOME/thesis/job_logs/run_topkSTM.log
python $HOME/thesis/propagate_with_topkSTM.py \
            --data_dir $TMPDIR/data \
            --log_dir $TMPDIR/output_dir \
            --model_path $TMPDIR/weights #/propagation_model.pth
echo "End: $(date)" >> $HOME/thesis/job_logs/run_topkSTM.log

#Copy output directory from scratch to home
cp -RT $TMPDIR/output_dir $HOME/thesis/results/topkSTM
