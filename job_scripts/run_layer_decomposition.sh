#!/bin/bash 

#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=gtx1080ti:4

# load modules
module load 2020
module load Python/3.8.2-GCCcore-9.3.0

# install dependencies
pip install --user --upgrade torch && pip install --user --upgrade torchvision

#Copy input file to scratch
VIDEO='kruispunt_rijks'
cp -RT $HOME/thesis/datasets/Videos/Images/$VIDEO $TMPDIR/video
mkdir $TMPDIR/00
cp -RT $HOME/thesis/datasets/Videos/Annotations/$VIDEO/00/00006.png $TMPDIR/00/00006.png
mkdir $TMPDIR/01
cp -RT $HOME/thesis/datasets/Videos/Annotations/$VIDEO/01/00006.png $TMPDIR/01/00006.png
mkdir $TMPDIR/weights
cp $HOME/thesis/models/third_party/weights/topkstm.pth $TMPDIR/weights/propagation_model.pth
cp $HOME/thesis/models/third_party/weights/raft.pth $TMPDIR/weights/flow_model.pth
cp $HOME/thesis/models/third_party/weights/monodepth.pth $TMPDIR/weights/depth_model.pth

#Create output directory on scratch
mkdir $TMPDIR/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
echo "$SLURM_JOBID | Start: $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log
python $HOME/thesis/run_layer_decomposition.py \
            --model_type 3d_bottleneck \
            --device cuda \
            --img_dir $TMPDIR/video \
            --initial_mask $TMPDIR/00/00006.png \
            --out_dir $TMPDIR/output_dir \
            --propagation_model $TMPDIR/weights/propagation_model.pth \
            --flow_model $TMPDIR/weights/flow_model.pth \
            --depth_model $TMPDIR/weights/depth_model.pth \
            --batch_size 4 \
            --n_epochs 700 \
            --save_freq 50 \
            --conv_channels 64 \
            --keydim 128 \
            --valdim 256 \
            --timesteps 4 \
            --num_context_frames 9 \
            --description 'Test final setup'

echo "$SLURM_JOBID | End:   $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log

#Copy output directory from scratch to home
mkdir -p $HOME/thesis/results/layer_decomposition/${VIDEO}_${SLURM_JOBID}
cp -RT $TMPDIR/output_dir $HOME/thesis/results/layer_decomposition/${VIDEO}_${SLURM_JOBID}
