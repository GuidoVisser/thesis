from argparse import ArgumentParser
from random import choices
from torch.utils.data import DataLoader
import torch
from os import path
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter

from InputProcessing.inputProcessor import InputProcessor
from models.DynamicLayerDecomposition.layerDecomposition import LayerDecompositer
from models.DynamicLayerDecomposition.loss_functions import DecompositeLoss2D, DecompositeLoss3D
from models.DynamicLayerDecomposition.layer_decomposition_networks import *

from utils.demo import create_decomposite_demo
from utils.utils import create_dir, seed_all
from models.DynamicLayerDecomposition.model_config import *


def init_dataloader(args, separate_bg, use_3d):
    """
    Initialize the input processor
    """

    input_processor = InputProcessor(
        args.model_type,
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
        use_3d=use_3d,
        separate_bg=separate_bg,
        frame_size=(args.frame_width, args.frame_height),
        do_jitter=True, 
        jitter_rate=args.jitter_rate,
        gt_in_memory=args.gt_in_memory
    )

    dataloader = DataLoader(
        input_processor, 
        batch_size=args.batch_size,
        shuffle=True
    )

    return dataloader

def init_loss_module(args):
    """
    Initialize the loss module
    """

    if args.model_type in ['2d', 'omnimatte']:
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
    else:
        loss_module = DecompositeLoss3D(
            args.lambda_mask,
            args.lambda_recon_flow,
            args.lambda_alpha_l0,
            args.lambda_alpha_l1,
            args.lambda_stabilization,
            args.lambda_dynamics_reg_diff,
            args.lambda_dynamics_reg_corr
        )

    return loss_module

def init_model(args, dataloader, loss_module, writer, separate_bg):
    """
    Initialize the decomposition model
    """

    if args.model_type == "2d":
        network = LayerDecompositionAttentionMemoryNet2D(
            dataloader.dataset.flow_handler.max_value,
            in_channels=args.in_channels,
            conv_channels=args.conv_channels,
            valdim=args.valdim,
            keydim=args.keydim,
            do_adjustment=True, 
            max_frames=len(dataloader.dataset.frame_iterator),
            coarseness=args.coarseness,
            shared_encoder=args.shared_encoder
            )
    elif args.model_type == "3d":
        network = LayerDecompositionAttentionMemoryNet3D(
            in_channels=args.in_channels,
            conv_channels=args.conv_channels,
            valdim=args.valdim,
            keydim=args.keydim,
            do_adjustment=True, 
            max_frames=len(dataloader.dataset.frame_iterator),
            coarseness=args.coarseness,
            shared_encoder=args.shared_encoder,
            mem_freq=args.mem_freq,
            timesteps=args.timesteps
        )
    elif args.model_type == "combined":
        network = LayerDecompositionAttentionMemoryNetCombined(
            in_channels=args.in_channels,
            conv_channels=args.conv_channels,
            valdim=args.valdim,
            keydim=args.keydim,
            do_adjustment=True, 
            max_frames=len(dataloader.dataset.frame_iterator),
            coarseness=args.coarseness,
            timesteps=args.timesteps,
            mem_freq=args.mem_freq,
            gt_in_memory=args.gt_in_memory
        )
    elif args.model_type == "3d_bottleneck":
        network = LayerDecompositionAttentionMemoryNet3DBottleneck(
            in_channels=args.in_channels,
            conv_channels=args.conv_channels,
            valdim=args.valdim,
            keydim=args.keydim,
            do_adjustment=True, 
            max_frames=len(dataloader.dataset.frame_iterator),
            coarseness=args.coarseness,
            shared_encoder=args.shared_encoder,
            gt_in_memory=args.gt_in_memory
        )
    else:
        network = Omnimatte(
            in_channels=args.in_channels,
            conv_channels=args.conv_channels,
            max_frames=len(dataloader.dataset.frame_iterator),
            coarseness=args.coarseness,
            do_adjustment=True
        )

    if args.device != "cpu":
        network = DataParallel(network).to(args.device)

    if args.continue_from != "":
        network.load_state_dict(torch.load(f"{args.continue_from}/reconstruction_weights.pth"))

    model = LayerDecompositer(
        dataloader, 
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

    return model

def main(args, start_epoch):

    # set seeds
    seed_all(args.seed)

    # set parameters
    if args.model_type == "3d":
        separate_bg = False
    else:
        separate_bg = True

    if args.model_type in ["2d", "omnimatte"]:
        args.timesteps = 2
        use_3d = False
    elif args.model_type == "3d_bottleneck":
        args.timesteps = 4
        use_3d = True
    else:
        args.timesteps = 16
        use_3d = True

    if args.model_type == "omnimatte":
        args.num_static_channels = args.in_channels

    # initialize components
    writer = SummaryWriter(args.out_dir)
    dataloader = init_dataloader(args, separate_bg, use_3d)
    loss_module = init_loss_module(args)
    model = init_model(args, dataloader, loss_module, writer, separate_bg)

    # run training
    model.run_training(start_epoch)

    # Set up for inference
    print("Epoch: final")
    dataloader.dataset.do_jitter = False
    
    # create final demo of results
    model.decomposite()
    create_decomposite_demo(path.join(args.out_dir, "decomposition/final"))

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
    directory_args.add_argument("--initial_mask", nargs="+", default=[f"datasets/{dataset}/Annotations/{video}/00000.png"], 
        help="paths to the initial object masks or the directories containing the object masks")
    directory_args.add_argument("--img_dir", type=str, default=f"datasets/{dataset}/JPEGImages/480p/{video}", 
        help="path to the directory in which the video frames are stored")
    directory_args.add_argument("--continue_from", type=str, default="", help="root directory of training run from which you wish to continue")

    model_args = parser.add_argument_group("model")
    model_args.add_argument("--model_type", type=str, default="3d_bottleneck", choices=["3d_bottleneck", "combined", "2d", "3d", "omnimatte"], help="The type of decomposition network to use")
    model_args.add_argument("--shared_encoder", action="store_true", help="Specifies whether to use a shared memory/query encoder in the network")
    model_args.add_argument("--conv_channels", type=int, default=16, help="base number of convolution channels in the convolutional neural networks")
    model_args.add_argument("--keydim", type=int, default=8, help="number of key channels in the attention memory network")
    model_args.add_argument("--valdim", type=int, default=16, help="number of value channels in the attention memory network")
    model_args.add_argument("--in_channels", type=int, default=16, help="number of channels in the input")
    model_args.add_argument("--coarseness", type=int, default=10, help="Temporal coarseness of camera adjustment parameters")

    input_args = parser.add_argument_group("model input")
    input_args.add_argument("--gt_in_memory", action="store_true", help="set true for ground truth memory input")
    input_args.add_argument("--num_static_channels", type=int, default=5, help="number of input channels that are static in time")
    input_args.add_argument("--timesteps", type=int, default=4, help="Temporal depth of the query input")
    input_args.add_argument("--mem_freq", type=int, default=1, help="period between frames that are added to memory")
    input_args.add_argument("--frame_height", type=int, default=256, help="target height of the frames")
    input_args.add_argument("--frame_width", type=int, default=448, help="target width of the frames")
    input_args.add_argument("--jitter_rate", type=float, default=0.75, help="rate of applying jitter to the input")
    input_args.add_argument("--composite_order", type=str, help="path to a text file containing the compositing order of the foreground objects")
    input_args.add_argument("--noise_temporal_coarseness", type=int, default=2, help="temporal coarseness of the dynamic noise input")

    training_param_args = parser.add_argument_group("training_parameters")
    training_param_args.add_argument("--batch_size", type=int, default=1, help="Batch size")
    training_param_args.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the reconstruction model")
    training_param_args.add_argument("--device", type=str, default="cuda", help="CUDA device")
    training_param_args.add_argument("--n_epochs", type=int, default=251, help="Number of epochs used for training")
    training_param_args.add_argument("--save_freq", type=int, default=70, help="Frequency at which the intermediate results are saved")
    training_param_args.add_argument("--n_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for training")
    training_param_args.add_argument("--seed", type=int, default=1, help="Random seed for libraries")
    training_param_args.add_argument("--alpha_bootstr_rolloff", type=int, default=5, help="Number of epochs to use mask bootstrap loss")
    training_param_args.add_argument("--alpha_loss_l1_rolloff", type=int, default=100, help="Number of epochs to use mask l1 regularization loss")

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
    if args.continue_from != "":
        CONFIG = load_config(f"{args.continue_from}/config.txt")

        # initialize epoch count
        start_epoch = CONFIG["training_parameters"]["n_epochs"] + 1
        CONFIG["training_parameters"]["n_epochs"] += args.n_epochs
        
        # update namespace
        args = read_config(args, CONFIG)
        save_config(f"{args.out_dir}/config.txt", CONFIG)
    else:
        create_dir(args.out_dir)
        start_epoch = 0
        CONFIG = update_config(args, CONFIG)
        save_config(f"{args.out_dir}/config.txt", CONFIG)
    

    main(args, start_epoch)
    print("done")