#!/bin/bash

#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=gtx1080ti:4

# get start time of script
DT = date +"%D_%T"

# load modules
module load 2020
module load Python

# instasll dependencies
pip install --user --upgrade torch && pip install --user --upgrade torchvision

#Copy input file to scratch
VIDEO="tennis"
cp -RT $HOME/thesis/datasets/DAVIS_sample/JPEGImages/480p/$VIDEO $TMPDIR/video
cp -RT $HOME/thesis/datasets/DAVIS_sample/Annotations/480p/$VIDEO/00000.png $TMPDIR/00000.png
mkdir $TMPDIR/weights
cp $HOME/thesis/models/third_party/weights/topkstm.pth $TMPDIR/weights/propagation_model.pth
cp $HOME/thesis/models/third_party/weights/raft.pth $TMPDIR/weights/flow_model.pth

#Create output directory on scratch
mkdir $TMPDIR/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
echo "Start: $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log
python $HOME/thesis/run_layer_decomposition.py \
            --img_dir $TMPDIR/video \
            --initial_mask $TMPDIR/00000.png \
            --out_dir $TMPDIR/output_dir \
            --propagation_model $TMPDIR/weights/propagation_model.pth \
            --flow_model $TMPDIR/weights/flow_model.pth \
            --batch_size 12 \
            --n_epochs 1
echo "End: $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log

#Copy output directory from scratch to home
mkdir -p $HOME/thesis/results/layer_decomposition/$VIDEO_$DT
cp -RT $TMPDIR/output_dir $HOME/thesis/results/layer_decomposition/$VIDEO_$DT
