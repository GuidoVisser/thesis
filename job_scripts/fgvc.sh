#!/bin/bash 

#SBATCH -n 1
#SBATCH -t 4:00:00
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
cp $HOME/thesis/models/weights/zip_serialization_false/edge_completion.pth $TMPDIR/weights/edge_completion.pth
cp $HOME/thesis/models/weights/zip_serialization_false/raft-things.pth $TMPDIR/weights/raft.pth
cp $HOME/thesis/models/weights/zip_serialization_false/imagenet_deepfill.pth $TMPDIR/weights/imagenet_deepfill.pth

#Create output directory on scratch
mkdir $TMPDIR/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
echo "$SLURM_JOBID | Start: $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log
python $HOME/thesis/fgvc.py \
        --data_dir $TMPDIR/video \
        --mask_dir $TMPDIR/masks \
        --outroot $TMPDIR/output_dir \
        --RAFT_weights $TMPDIR/weights/raft.pth \
        --deepfill_model $TMPDIR/weights/imagenet_deepfill.pth \
        --edge_completion_model $TMPDIR/weights/edge_completion.pth \
        --seamless

echo "$SLURM_JOBID | End:   $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log

#Copy output directory from scratch to home
mkdir -p $HOME/thesis/results/fgvc/${VIDEO}_${SLURM_JOBID}
cp -RT $TMPDIR/output_dir $HOME/thesis/results/fgvc/${VIDEO}_${SLURM_JOBID}
