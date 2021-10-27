#!/bin/bash

#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=gtx1080ti:4

# get start time of script
DT=`date +"%m_%d_%H_%M_%S"`

# load modules
module load 2020
module load Python

# instasll dependencies
pip install --user --upgrade torch && pip install --user --upgrade torchvision

#Copy input file to scratch
VIDEO="scooter-black"
cp -RT $HOME/thesis/datasets/DAVIS_minisample/JPEGImages/480p/$VIDEO $TMPDIR/video
cp -RT $HOME/thesis/datasets/DAVIS_minisample/Annotations/480p/$VIDEO/00000.png $TMPDIR/00000.png
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
            --n_epochs 2001 \
            --save_freq 100 \
            --mem_freq 2 \
            --alpha_bootstr_thresh 5e-5 \
            --experiment_config 3 \
            --description 'Dynamic model with 2001 epochs and high memory frequency. TopkSTM pretrained backbones are used for the memory backbones with channels for object masks included. alpha_bootstrap_threshold is set low. The context is added to the input of the decoder of the reconstruction UNet in the channel dimension. \n    Experiment 3: Only the bg layer has a memory network. A 1x1 convolutional kernel is used to downsample the channels at the point where the context is added.'
echo "End: $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log

#Copy output directory from scratch to home
mkdir -p $HOME/thesis/results/layer_decomposition/${VIDEO}_${DT}
cp -RT $TMPDIR/output_dir $HOME/thesis/results/layer_decomposition/${VIDEO}_${DT}
