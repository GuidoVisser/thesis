from argparse import ArgumentParser
from cv2 import NORM_MINMAX
from torch._C import device
from torch.utils.data import DataLoader
import torch
import numpy as np
from os import path
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter

from InputProcessing.inputProcessor import InputProcessor
from models.DynamicLayerDecomposition.layerDecomposition import LayerDecompositer
from models.DynamicLayerDecomposition.loss_functions import DecompositeLoss2D
from models.DynamicLayerDecomposition.modules.modules_2d import LayerDecompositionAttentionMemoryNet2D

from utils.demo import create_decomposite_demo
from utils.utils import create_dir, seed_all
from models.DynamicLayerDecomposition.model_config import CONFIG, update_config, save_config 


def main(args):

    seed_all(args.seed)

    writer = SummaryWriter(args.out_dir)
    separate_bg = True

    input_processor = InputProcessor(
        args.img_dir, 
        args.out_dir, 
        args.initial_mask, 
        args.composite_order, 
        propagation_model=args.propagation_model, 
        flow_model=args.flow_model,
        in_channels=args.in_channels,
        num_static_channels=args.num_static_channels,
        noise_temporal_coarseness=args.noise_temporal_coarseness,
        device=args.device,
        timesteps=args.timesteps,
        use_3d=False,
        separate_bg=separate_bg,
        frame_size=(args.frame_width, args.frame_height),
        do_jitter=True, 
        jitter_rate=args.jitter_rate
    )

    data_loader = DataLoader(
        input_processor, 
        batch_size=args.batch_size,
        shuffle=True
    )

    loss_module = DecompositeLoss2D(
        args.lambda_mask,
        args.lambda_recon_flow,
        args.lambda_recon_warp,
        args.lambda_alpha_warp,
        args.lambda_alpha_l0,
        args.lambda_alpha_l1,
        args.lambda_stabilization,
        args.lambda_dynamics_reg_diff,
        args.lambda_dynamics_reg_corr
    )

    network = DataParallel(LayerDecompositionAttentionMemoryNet2D(
        in_channels=args.in_channels,
        conv_channels=args.conv_channels,
        do_adjustment=True, 
        max_frames=len(input_processor) + args.timesteps,
        coarseness=args.coarseness,
        shared_encoder=bool(args.shared_encoder)
    )).to(args.device)

    model = LayerDecompositer(
        data_loader, 
        loss_module, 
        network, 
        writer,
        args.learning_rate, 
        args.alpha_bootstr_rolloff,
        args.alpha_loss_l1_rolloff,
        results_root=args.out_dir, 
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        save_freq=args.save_freq,
        separate_bg=separate_bg
    )

    model.run_training()

    # Set up for inference
    print("Epoch: inference")
    input_processor.do_jitter = False
    
    model.decomposite()
    create_decomposite_demo(path.join(args.out_dir, "decomposition/inference"))

if __name__ == "__main__":
    print("started")
    print(f"Running on {torch.cuda.device_count()} GPU{'s' if torch.cuda.device_count() > 1 else ''}")
    parser = ArgumentParser()
    parser.add_argument("--description", type=str, default="no description given", help="description of the experiment")

    dataset = "DAVIS_minisample"
    video = "scooter-black"
    directory_args = parser.add_argument_group("directories")
    directory_args.add_argument("--out_dir", type=str, default=f"results/layer_decomposition_dynamic/{video}", 
        help="path to directory where results are saved")
    directory_args.add_argument("--initial_mask", type=str, default=f"datasets/{dataset}/Annotations/480p/{video}/00000.png", 
        help="path to the initial mask")
    directory_args.add_argument("--img_dir", type=str, default=f"datasets/{dataset}/JPEGImages/480p/{video}", 
        help="path to the directory in which the video frames are stored")
    
    reconstruction_model_args = parser.add_argument_group("reconstruction_model")
    reconstruction_model_args.add_argument("--composite_order", type=str, 
        help="path to a text file containing the compositing order of the foreground objects")
    reconstruction_model_args.add_argument("--coarseness", type=int, default=10, 
        help="Temporal coarseness of camera adjustment parameters")

    memory_network_args = parser.add_argument_group("memory_network")
    memory_network_args.add_argument("--keydim", type=int, default=8, help="number of key channels in the attention memory network")
    memory_network_args.add_argument("--valdim", type=int, default=16, help="number of value channels in the attention memory network")

    training_param_args = parser.add_argument_group("training_parameters")
    training_param_args.add_argument("--batch_size", type=int, default=1, help="Batch size")
    training_param_args.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the reconstruction model")
    training_param_args.add_argument("--device", type=str, default="cuda", help="CUDA device")
    training_param_args.add_argument("--n_epochs", type=int, default=1, help="Number of epochs used for training")
    training_param_args.add_argument("--save_freq", type=int, default=50, help="Frequency at which the intermediate results are saved")
    training_param_args.add_argument("--n_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for training")
    training_param_args.add_argument("--seed", type=int, default=1, help="Random seed for libraries")
    training_param_args.add_argument("--alpha_bootstr_rolloff", type=int, default=50, help="Number of epochs to use mask bootstrap loss")
    training_param_args.add_argument("--alpha_loss_l1_rolloff", type=int, default=100, help="Number of epochs to use mask l1 regularization loss")
    training_param_args.add_argument("--experiment_config", type=int, default=2, help="configuration id for the experiment that is being run")
    training_param_args.add_argument("--in_channels", type=int, default=16, help="number of channels in the input")
    training_param_args.add_argument("--num_static_channels", type=int, default=5, help="number of input channels that are static in time")
    training_param_args.add_argument("--conv_channels", type=int, default=16, help="base number of convolution channels in the convolutional neural networks")
    training_param_args.add_argument("--noise_temporal_coarseness", type=int, default=2, help="temporal coarseness of the dynamic noise input")
    training_param_args.add_argument("--shared_encoder", type=int, default=1, help="Specifies whether to use a shared memory/query encoder in the network")
    training_param_args.add_argument("--timesteps", type=int, default=2, help="Temporal depth of the query input")
    training_param_args.add_argument("--frame_height", type=int, default=256, help="target height of the frames")
    training_param_args.add_argument("--frame_width", type=int, default=448, help="target width of the frames")
    training_param_args.add_argument("--jitter_rate", type=float, default=0.75, help="rate of applying jitter to the input")

    lambdas = parser.add_argument_group("lambdas")
    lambdas.add_argument("--lambda_mask", type=float, default=50., help="starting value for the lambda of the alpha_mask_bootstrap loss")
    lambdas.add_argument("--lambda_recon_flow", type=float, default=1., help="lambda of the flow reconstruction loss")
    lambdas.add_argument("--lambda_recon_warp", type=float, default=0., help="lambda of the warped rgb reconstruction loss")
    lambdas.add_argument("--lambda_alpha_warp", type=float, default=0.005, help="lambda of the warped alpha estimation loss")
    lambdas.add_argument("--lambda_alpha_l0", type=float, default=0.005, help="lambda of the l0 part of the alpha regularization loss")
    lambdas.add_argument("--lambda_alpha_l1", type=float, default=0.01, help="lambda of the l1 part of the alpha regularization loss")
    lambdas.add_argument("--lambda_stabilization", type=float, default=0.001, help="lambda of the camera stabilization loss")
    lambdas.add_argument("--lambda_dynamics_reg_diff", type=float, default=0.05, help="lambda of the difference part of the dynamics regularization loss")
    lambdas.add_argument("--lambda_dynamics_reg_corr", type=float, default=0.1, help="lambda of the correlation part of the dynamics regularization loss")

    pretrained_model_args = parser.add_argument_group("pretrained_models")
    pretrained_model_args.add_argument("--propagation_model", type=str, default="models/third_party/weights/propagation_model.pth", 
        help="path to the weights of the mask propagation model")
    pretrained_model_args.add_argument("--flow_model", type=str, default="models/third_party/weights/raft-things.pth",
        help="path to the optical flow estimation model")

    args = parser.parse_args()

    # update and save arguments
    create_dir(args.out_dir)
    CONFIG = update_config(args, CONFIG)
    save_config(f"{args.out_dir}/config.txt", CONFIG)

    main(args)
    print("done")