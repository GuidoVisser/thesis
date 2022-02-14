#!/bin/bash 

#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=gtx1080ti:1

# load modules
module load 2020
module load Python/3.8.2-GCCcore-9.3.0

# install dependencies
# pip install --user --upgrade torch && pip install --user --upgrade torchvision

#Copy input file to scratch
VIDEO='kruispunt_rijks'
cp -RT $HOME/thesis/datasets/Videos_small/Images/$VIDEO $TMPDIR/video
cp -RT $HOME/thesis/datasets/Videos_small/Annotations/$VIDEO/ $TMPDIR/masks
mkdir $TMPDIR/weights
cp $HOME/thesis/models/third_party/onion_peel/OPN.pth $TMPDIR/weights/OPN.pth
cp $HOME/thesis/models/third_party/onion_peel/TCN.pth $TMPDIR/weights/TCN.pth


#Create output directory on scratch
mkdir $TMPDIR/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
echo "$SLURM_JOBID | Start: $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log
python $HOME/thesis/onion_peel.py \
        --img_dir $TMPDIR/video \
        --mask_dir $TMPDIR/masks \
        --out_dir $TMPDIR/output_dir
echo "$SLURM_JOBID | End:   $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log

#Copy output directory from scratch to home
mkdir -p $HOME/thesis/results/onion_peel/${VIDEO}_${SLURM_JOBID}
cp -RT $TMPDIR/output_dir $HOME/thesis/results/onion_peel/${VIDEO}_${SLURM_JOBID}
