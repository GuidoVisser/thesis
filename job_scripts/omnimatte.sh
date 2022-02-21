#!/bin/bash 

#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=gtx1080ti:4

# load modules
module load 2020
module load Python/3.8.2-GCCcore-9.3.0

# install dependencies
pip install --user --upgrade torch && pip install --user --upgrade torchvision

#Copy input file to scratch
VIDEO='nescio_2'
cp -RT $HOME/thesis/datasets/Videos/Images/$VIDEO $TMPDIR/$VIDEO
mkdir $TMPDIR/masks
cp -RT $HOME/thesis/datasets/Videos/Annotations/$VIDEO $TMPDIR/masks
mkdir $TMPDIR/weights
cp $HOME/thesis/models/third_party/weights/topkstm.pth $TMPDIR/weights/propagation_model.pth
cp $HOME/thesis/models/third_party/weights/raft.pth $TMPDIR/weights/flow_model.pth
cp $HOME/thesis/models/third_party/weights/monodepth.pth $TMPDIR/weights/depth_model.pth

#Create output directory on scratch
mkdir $TMPDIR/output_dir

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
echo "$SLURM_JOBID | Start: $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log
python $HOME/thesis/run_layer_decomposition.py \
            --model_type omnimatte \
            --device cuda \
            --img_dir $TMPDIR/$VIDEO \
            --mask_dir $TMPDIR/masks \
            --out_dir $TMPDIR/output_dir \
            --propagation_model $TMPDIR/weights/propagation_model.pth \
            --flow_model $TMPDIR/weights/flow_model.pth \
            --depth_model $TMPDIR/weights/depth_model.pth \
            --use_2d_loss_module \
            --timesteps 2 \
            --num_static_channels 13 \
            --in_channels 16 \
            --lambda_mask 50.0 \
            --batch_size 8 \
            --n_epochs 1500 \
            --save_freq 500 \
            --conv_channels 64 \
            --description 'Omnimatte final results'

echo "$SLURM_JOBID | End:   $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log

#Copy output directory from scratch to home
mkdir -p $HOME/thesis/results/omnimatte/${VIDEO}_${SLURM_JOBID}
cp -RT $TMPDIR/output_dir $HOME/thesis/results/omnimatte/${VIDEO}_${SLURM_JOBID}
