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
from models.DynamicLayerDecomposition.model_config import default_config, read_config, load_config, save_config, update_config


class ExperimentRunner(object):
    """
    Object that initializes the model, dataloader and loss modules for different experiment setups
    
    Experiments are defined by the model setup and memory setup

    Model setup (default is 2D convolutions):
    1. Non strided 3D convolutions in memory encoder 
    2. strided 3D convolutions in memory encoder
    3. A single 3D convolution as bottleneck after the context is added
    4. A single 3D convolution as bottleneck after the context is added. The encoder share weights
    5. No 3D convolutions are added
    6. No 3D convolutions are added. The encoders share weights
    7. Omnimatte

    Memory setup:
    1. noise
    2. rgb
    3. noise + depth + flow
    4. rgb + depth + flow

    NOTE: Encoders cannot share weights when rgb is used in the memory input
    
    """
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        model_setup = args.model_setup
        memory_setup = args.memory_setup

        self._set_configs(model_setup, memory_setup)

        # update and save arguments   
        if args.continue_from != "":
            config = load_config(f"{args.continue_from}/config.txt")

            # initialize epoch count
            self.start_epoch = config["training_parameters"]["n_epochs"] + 1
            config["training_parameters"]["n_epochs"] += args.n_epochs
            
            # update namespace
            args = read_config(args, config)
            save_config(f"{args.out_dir}/config.txt", config)
        else:
            create_dir(args.out_dir)
            self.start_epoch = 0
            config = update_config(args, default_config())
            save_config(f"{args.out_dir}/config.txt", config)
    
    def _set_configs(self, model_setup: int, memory_setup: int) -> None:

        if model_setup == 1:
            self.args.model_type = "3d_memory"
        elif model_setup == 2:
            self.args.model_type = "3d_memory"
            self.args.memory_t_strided = True
            self.args.memory_timesteps = 16
        elif model_setup == 3:
            self.args.model_type = "3d_bottleneck"
        elif model_setup == 4:
            self.args.model_type = "3d_bottleneck"
            self.args.shared_backbone = True
        elif model_setup == 5:
            self.args.model_type = "fully_2d"            
            self.args.use_2d_loss_module = True
            self.args.timesteps = 2
        elif model_setup == 6:
            self.args.model_type = "fully_2d"   
            self.args.shared_backbone = True
            self.args.use_2d_loss_module = True
            self.args.timesteps = 2
        elif model_setup == 7:
            self.args.model_type = "omnimatte"
            self.args.timesteps = 2
            self.args.num_static_channels = self.args.in_channels
            self.args.use_2d_loss_module = True
            self.args.lambda_recon_depth = 0.

        if memory_setup == 1:
            self.args.memory_input_type = "noise"
            if self.args.shared_backbone:
                self.args.memory_in_channels = 16
            else:
                self.args.memory_in_channels = 12
        elif memory_setup == 2:
            self.args.memory_input_type = "rgb"
            self.args.memory_in_channels = 3
            if self.args.shared_backbone:
                raise ValueError("Cannot use shared encoder with rgb memory input")
        elif memory_setup == 3:
            self.args.memory_input_type = "noise_flow_depth"
            if self.args.shared_backbone:
                self.args.memory_in_channels = 16
            else:
                self.args.memory_in_channels = 15
        elif memory_setup == 4:
            self.args.memory_input_type = "rgb_flow_depth"
            self.args.memory_in_channels = 6
            if self.args.shared_backbone:
                raise ValueError("Cannot use shared encoder with rgb memory input")

    def start(self):
        # set seeds
        seed_all(self.args.seed)

        # initialize components
        writer = SummaryWriter(self.args.out_dir)
        dataloader = self.init_dataloader()
        loss_module = self.init_loss_module()
        model = self.init_model(dataloader, loss_module, writer)

        # run training
        model.run_training(self.start_epoch)

        # Set up for inference
        print("Epoch: final")
        dataloader.dataset.do_jitter = False
        
        # create final demo of results
        model.decomposite()
        create_decomposite_demo(path.join(self.args.out_dir, "decomposition/final"))

    def init_dataloader(self):
        """
        Initialize the input processor
        """

        input_processor = InputProcessor(
            self.args,
            do_jitter=True
        )

        dataloader = DataLoader(
            input_processor, 
            batch_size=args.batch_size,
            shuffle=True
        )

        return dataloader

    def init_loss_module(self):
        """
        Initialize the loss module
        """

        if self.args.use_2d_loss_module:
            loss_module = DecompositeLoss2D(
                self.args.lambda_mask,
                self.args.lambda_recon_flow,
                self.args.lambda_recon_depth,
                self.args.lambda_recon_warp,
                self.args.lambda_alpha_warp,
                self.args.lambda_alpha_l0,
                self.args.lambda_alpha_l1,
                self.args.lambda_stabilization,
                self.args.lambda_dynamics_reg_diff,
                self.args.lambda_dynamics_reg_corr
            )
        else:
            loss_module = DecompositeLoss3D(
                self.args.lambda_mask,
                self.args.lambda_recon_flow,
                self.args.lambda_recon_depth,
                self.args.lambda_alpha_l0,
                self.args.lambda_alpha_l1,
                self.args.lambda_stabilization,
                self.args.lambda_dynamics_reg_diff,
                self.args.lambda_dynamics_reg_corr
            )
        return loss_module

    def init_model(self, dataloader, loss_module, writer):
        """
        Initialize the decomposition model
        """

        if self.args.model_type == "3d_bottleneck":
            network = LayerDecompositionAttentionMemoryNet3DBottleneck(
                in_channels=self.args.in_channels,
                memory_in_channels=self.args.memory_in_channels,
                shared_backbone=self.args.shared_backbone,
                conv_channels=self.args.conv_channels,
                valdim=self.args.valdim,
                keydim=self.args.keydim,
                topk=self.args.topk,
                do_adjustment=True, 
                max_frames=len(dataloader.dataset.frame_iterator),
                coarseness=self.args.coarseness
            )    
        elif self.args.model_type == "3d_memory":
            network = LayerDecompositionAttentionMemoryNet3DMemoryEncoder(
                in_channels=self.args.in_channels,
                memory_in_channels=self.args.memory_in_channels,
                t_strided=self.args.memory_t_strided,
                conv_channels=self.args.conv_channels,
                valdim=self.args.valdim,
                keydim=self.args.keydim,
                topk=self.args.topk,
                do_adjustment=True, 
                max_frames=len(dataloader.dataset.frame_iterator),
                coarseness=self.args.coarseness,
            )     
        elif self.args.model_type == "omnimatte":
            network = Omnimatte(
                in_channels=self.args.in_channels,
                conv_channels=args.conv_channels,
                max_frames=len(dataloader.dataset.frame_iterator),
                coarseness=args.coarseness,
                do_adjustment=True
            )
        elif self.args.model_type == "fully_2d":
            network = LayerDecompositionAttentionMemoryNet2D(
                in_channels=self.args.in_channels,
                memory_in_channels=self.args.memory_in_channels,
                shared_backbone=self.args.shared_backbone,
                flow_max=dataloader.dataset.flow_handler.max_value,
                conv_channels=self.args.conv_channels,
                valdim=self.args.valdim,
                keydim=self.args.keydim,
                topk=self.args.topk,
                do_adjustment=True, 
                max_frames=len(dataloader.dataset.frame_iterator),
                coarseness=self.args.coarseness
            )   

        if self.args.device != "cpu":
            network = DataParallel(network).to(args.device)

        if self.args.continue_from != "":
            network.load_state_dict(torch.load(f"{args.continue_from}/reconstruction_weights.pth"))

        model = LayerDecompositer(
            dataloader, 
            loss_module, 
            network, 
            writer,
            self.args.learning_rate, 
            self.args.alpha_bootstr_rolloff,
            self.args.alpha_loss_l1_rolloff,
            results_root=self.args.out_dir, 
            batch_size=self.args.batch_size,
            n_epochs=self.args.n_epochs,
            save_freq=self.args.save_freq,
            separate_bg=self.args.no_static_background
        )

        return model


if __name__ == "__main__":
    print("started")
    print(f"Running on {torch.cuda.device_count()} GPU{'s' if torch.cuda.device_count() > 1 else ''}")
    parser = ArgumentParser()
    parser.add_argument("--description", type=str, default="no description given", help="description of the experiment")
    parser.add_argument("--model_setup", type=int, default=4, help="id of model setup")
    parser.add_argument("--memory_setup", type=int, default=3, help="id of memory input setup")

    dataset = "DAVIS_minisample"
    video = "scooter-black"
    directory_args = parser.add_argument_group("directories")
    directory_args.add_argument("--out_dir", type=str, default=f"results/layer_decomposition_dynamic/{video}", 
        help="path to directory where results are saved")
    directory_args.add_argument("--initial_mask", nargs="+", default=[f"datasets/{dataset}/Annotations/480p/{video}/00000.png"], 
        help="paths to the initial object masks or the directories containing the object masks")
    directory_args.add_argument("--img_dir", type=str, default=f"datasets/{dataset}/JPEGImages/480p/{video}", 
        help="path to the directory in which the video frames are stored")
    directory_args.add_argument("--continue_from", type=str, default="", help="root directory of training run from which you wish to continue")

    model_args = parser.add_argument_group("model")
    model_args.add_argument("--model_type", type=str, default="3d_bottleneck", choices=["3d_bottleneck", "3d_memory", "fully_2d", "fully_3d", "omnimatte"], help="The type of decomposition network to use")
    model_args.add_argument("--conv_channels", type=int, default=16, help="base number of convolution channels in the convolutional neural networks")
    model_args.add_argument("--keydim", type=int, default=8, help="number of key channels in the attention memory network")
    model_args.add_argument("--valdim", type=int, default=16, help="number of value channels in the attention memory network")
    model_args.add_argument("--in_channels", type=int, default=16, help="number of channels in the input")
    model_args.add_argument("--memory_in_channels", type=int, default=16, help="number of channels in the memory input")
    model_args.add_argument("--coarseness", type=int, default=10, help="Temporal coarseness of camera adjustment parameters")
    model_args.add_argument("--use_2d_loss_module", action="store_true", help="Use 2d loss module in stead of 3d loss module")
    model_args.add_argument("--no_static_background", action="store_true", help="Don't use separated static and dynamic background")
    model_args.add_argument("--shared_backbone", action="store_true", help="backbones for query and memory encoder have shared weight")
    model_args.add_argument("--memory_t_strided", action="store_true", help="If 3D convolutions are used in memory encoders, set them to be strided in time dimension")
    model_args.add_argument("--topk", type=int, default=0, help="k value for topk channel selection in context distribution")

    input_args = parser.add_argument_group("model input")
    input_args.add_argument("--num_static_channels", type=int, default=5, help="number of input channels that are static in time")
    input_args.add_argument("--timesteps", type=int, default=4, help="Temporal depth of the query input")
    input_args.add_argument("--memory_timesteps", type=int, default=4, help="Temporal depth of the memory input")
    input_args.add_argument("--mem_freq", type=int, default=1, help="period between frames that are added to memory")
    input_args.add_argument("--frame_height", type=int, default=256, help="target height of the frames")
    input_args.add_argument("--frame_width", type=int, default=448, help="target width of the frames")
    input_args.add_argument("--jitter_rate", type=float, default=0.75, help="rate of applying jitter to the input")
    input_args.add_argument("--composite_order", type=str, help="path to a text file containing the compositing order of the foreground objects")
    input_args.add_argument("--noise_temporal_coarseness", type=int, default=2, help="temporal coarseness of the dynamic noise input")
    input_args.add_argument("--memory_input_type", type=str, default="noise_flow_depth", choices=["rgb", "rgb_flow_depth", "noise", "noise_flow_depth"])

    training_param_args = parser.add_argument_group("training_parameters")
    training_param_args.add_argument("--batch_size", type=int, default=1, help="Batch size")
    training_param_args.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the reconstruction model")
    training_param_args.add_argument("--device", type=str, default="cuda", help="CUDA device")
    training_param_args.add_argument("--n_epochs", type=int, default=251, help="Number of epochs used for training")
    training_param_args.add_argument("--save_freq", type=int, default=70, help="Frequency at which the intermediate results are saved")
    training_param_args.add_argument("--n_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for training")
    training_param_args.add_argument("--seed", type=int, default=1, help="Random seed for libraries")

    lambdas = parser.add_argument_group("lambdas")
    lambdas.add_argument("--lambda_mask", nargs="+", default=[50., 50, 0.], help="values for the lambda of the alpha_mask_bootstrap loss")
    lambdas.add_argument("--lambda_recon_flow", nargs="+", default=[1.], help="lambda of the flow reconstruction loss")
    lambdas.add_argument("--lambda_recon_warp", nargs="+", default=[0.], help="lambda of the warped rgb reconstruction loss")
    lambdas.add_argument("--lambda_recon_depth", nargs="+", default=[0.5], help="lambda of the depth reconstruction loss")
    lambdas.add_argument("--lambda_alpha_warp", nargs="+", default=[0.005], help="lambda of the warped alpha estimation loss")
    lambdas.add_argument("--lambda_alpha_l0", nargs="+", default=[0.005], help="lambda of the l0 part of the alpha regularization loss")
    lambdas.add_argument("--lambda_alpha_l1", nargs="+", default=[0.01, 100, 0.], help="lambda of the l1 part of the alpha regularization loss")
    lambdas.add_argument("--lambda_stabilization", nargs="+", default=[0.001], help="lambda of the camera stabilization loss")
    lambdas.add_argument("--lambda_dynamics_reg_diff", nargs="+", default=[0.005], help="lambda of the difference part of the dynamics regularization loss")
    lambdas.add_argument("--lambda_dynamics_reg_corr", nargs="+", default=[0.01], help="lambda of the correlation part of the dynamics regularization loss")

    pretrained_model_args = parser.add_argument_group("pretrained_models")
    pretrained_model_args.add_argument("--propagation_model", type=str, default="models/third_party/weights/propagation_model.pth", 
        help="path to the weights of the mask propagation model")
    pretrained_model_args.add_argument("--flow_model", type=str, default="models/third_party/weights/raft-things.pth",
        help="path to the weights of the optical flow estimation model")
    pretrained_model_args.add_argument("--depth_model", type=str, default="models/third_party/weights/monodepth_resnet18_001.pth",
        help="path to the weights of the depth estimation model")

    args = parser.parse_args()

    experiment_runner = ExperimentRunner(args)
    experiment_runner.start()

    print("done")