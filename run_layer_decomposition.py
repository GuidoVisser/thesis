from argparse import ArgumentParser
from datetime import datetime
from torch.utils.data import DataLoader
import torch
import json
from os import path, listdir
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter

from InputProcessing.inputProcessor import ContextDataset, InputProcessor
from models.DynamicLayerDecomposition.layerDecomposition import LayerDecompositer
from models.DynamicLayerDecomposition.loss_functions import DecompositeLoss2D, DecompositeLoss3D
from models.DynamicLayerDecomposition.layer_decomposition_networks import *

from InputProcessing.backgroundVolume import BackgroundVolume
from InputProcessing.maskHandler import MaskHandler
from InputProcessing.flowHandler import FlowHandler
from InputProcessing.frameIterator import FrameIterator
from InputProcessing.homography import HomographyHandler
from InputProcessing.depthHandler import DepthHandler

from utils.demo import create_decomposite_demo
from utils.utils import create_dir, seed_all, create_dirs

import cv2

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

        self._set_configs()

        # update and save arguments   
        if args.continue_from != "":
            new_epochs = args.n_epochs
            with open(f"{args.out_dir}/config.json", "r") as f:
                args.__dict__ = json.load(f)

            self.start_epoch = args.n_epochs + 1
            args.n_epochs += new_epochs

            with open(f"{args.out_dir}/config.json", "w") as f:
                json.dump(args.__dict__, f, indent=2)

        else:
            create_dir(args.out_dir)
            self.start_epoch = 0

            with open(f"{args.out_dir}/config.json", "w") as f:
                json.dump(args.__dict__, f, indent=2)
            
        with open(f"{args.out_dir}/config.txt", "w") as f:
            for arg in vars(args):
                f.write(f"{arg}{' '*(30 - len(arg))}{getattr(args, arg)}\n")
        
    
    def _set_configs(self) -> None:

        if self.args.model_type == "fully_2d":
            self.args.use_2d_loss_module = True
            self.args.timesteps = 2
        elif self.args.model_type == "no_addons":
            self.args.use_2d_loss_module = True
            self.args.timesteps = 2
            self.args.lambda_alpha_warp = [0.]
        elif self.args.model_type == "omnimatte":
            self.args.use_2d_loss_module = True
            self.args.timesteps = 2
            self.args.num_static_channels = self.args.in_channels
            self.args.lambda_recon_depth = [0.]
            self.args.use_depth = False

    def start(self):
        # set seeds
        seed_all(self.args.seed)

        # initialize components
        writer = SummaryWriter(self.args.out_dir)
        dataloader, context_loader = self.init_dataloader()
        loss_module = self.init_loss_module()
        model = self.init_model(dataloader, context_loader, loss_module, writer)

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

        img_dir        = path.join(self.args.out_dir, "images")
        mask_dir       = path.join(self.args.out_dir, "masks")
        flow_dir       = path.join(self.args.out_dir, "flow")
        depth_dir      = path.join(self.args.out_dir, "depth")
        background_dir = path.join(self.args.out_dir, "background")
        create_dirs(img_dir, mask_dir, flow_dir, depth_dir, background_dir)

        # prepare image directory
        frame_paths = [frame for frame in sorted(listdir(self.args.img_dir))]
        for i, frame_path in enumerate(frame_paths):
            img = cv2.resize(cv2.imread(path.join(self.args.img_dir, frame_path)), (self.args.frame_width, self.args.frame_height), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(path.join(img_dir, f"{i:05}.jpg"), img)

        # create helper classes 
        #   These helpers prepare the mask propagation, homography estimation and optical flow calculation 
        #   at initialization and save the results for fast retrieval
        frame_iterator     = FrameIterator(img_dir, self.args)
        mask_handler       = MaskHandler(mask_dir, self.args)
        flow_handler       = FlowHandler(frame_iterator, mask_handler, flow_dir, raft_weights=self.args.flow_model, device=self.args.device, iters=50)
        homography_handler = HomographyHandler(self.args.out_dir, img_dir, self.args.device, (self.args.frame_width, self.args.frame_height))
        background_volume  = BackgroundVolume(background_dir, homography_handler, self.args)
        background_volume.visualize(0)
        if self.args.use_depth:
            depth_handler  = DepthHandler(img_dir, depth_dir, self.args, mask_handler)
        else:
            depth_handler = None

        input_processor = InputProcessor(
            self.args,
            frame_iterator,
            mask_handler,
            flow_handler,
            homography_handler,
            depth_handler,
            background_volume,
            do_jitter=True
        )

        dataloader = DataLoader(
            input_processor, 
            batch_size=args.batch_size,
            shuffle=True
        )

        context_loader = ContextDataset(
            self.args,
            frame_iterator,
            mask_handler,
            flow_handler,
            homography_handler,
            depth_handler,
            background_volume
        )

        return dataloader, context_loader

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
                self.args.lambda_dynamics_reg_corr,
                self.args.lambda_dynamics_reg_diff,
                self.args.lambda_dynamics_reg_l0,
                self.args.lambda_dynamics_reg_l1,
                self.args.lambda_detail_reg,
                self.args.lambda_bg_scaling,
                self.args.corr_diff,
                self.args.alpha_reg_layers,
                self.args.use_alpha_detail_reg
            )
        else:
            loss_module = DecompositeLoss3D(
                self.args.lambda_mask,
                self.args.lambda_recon_flow,
                self.args.lambda_recon_depth,
                self.args.lambda_alpha_l0,
                self.args.lambda_alpha_l1,
                self.args.lambda_stabilization,
                self.args.lambda_dynamics_reg_corr,
                self.args.lambda_dynamics_reg_diff,
                self.args.lambda_dynamics_reg_l0,
                self.args.lambda_dynamics_reg_l1,
                self.args.lambda_detail_reg,
                self.args.lambda_bg_scaling,
                self.args.corr_diff,
                self.args.alpha_reg_layers,
                self.args.use_alpha_detail_reg
            )
        return loss_module

    def init_model(self, dataloader, context_loader, loss_module, writer):
        """
        Initialize the decomposition model
        """

        if self.args.model_type == "3d_bottleneck":
            if not self.args.use_depth:
                network = LayerDecompositionAttentionMemoryNet3DBottleneck(
                    context_loader=context_loader,
                    num_context_frames=args.num_context_frames,
                    in_channels=self.args.in_channels,
                    conv_channels=self.args.conv_channels,
                    valdim=self.args.valdim,
                    keydim=self.args.keydim,
                    topk=self.args.topk,
                    do_adjustment=True, 
                    max_frames=len(dataloader.dataset.frame_iterator),
                    transposed_bottleneck=not self.args.bottleneck_normal,
                    br_coarseness=self.args.br_coarseness,
                    offset_coarseness=self.args.offset_coarseness,
                    separate_value_layer=self.args.separate_value_layer,
                    unsampled_dynamic_bg_input=self.args.unsampled_dynamic_bg_input
                )
            else:
                network = LayerDecompositionAttentionMemoryDepthNet3DBottleneck(
                    context_loader,
                    num_context_frames=args.num_context_frames,
                    in_channels=self.args.in_channels,
                    conv_channels=self.args.conv_channels,
                    valdim=self.args.valdim,
                    keydim=self.args.keydim,
                    topk=self.args.topk,
                    do_adjustment=True, 
                    max_frames=len(dataloader.dataset.frame_iterator),
                    transposed_bottleneck=not self.args.bottleneck_normal,
                    br_coarseness=self.args.br_coarseness,
                    offset_coarseness=self.args.offset_coarseness,
                    separate_value_layer=self.args.separate_value_layer
                )    
        elif self.args.model_type == "omnimatte":
            network = Omnimatte(
                in_channels=self.args.in_channels,
                conv_channels=args.conv_channels,
                max_frames=len(dataloader.dataset.frame_iterator),
                br_coarseness=self.args.br_coarseness,
                offset_coarseness=self.args.offset_coarseness,
                do_adjustment=True
            )
        elif self.args.model_type == "fully_2d":
            if not self.args.use_depth:
                network = LayerDecompositionAttentionMemoryNet2D(
                    context_loader,
                    num_context_frames=args.num_context_frames,
                    in_channels=self.args.in_channels,
                    conv_channels=self.args.conv_channels,
                    valdim=self.args.valdim,
                    keydim=self.args.keydim,
                    topk=self.args.topk,
                    do_adjustment=True, 
                    max_frames=len(dataloader.dataset.frame_iterator),
                    br_coarseness=self.args.br_coarseness,
                    offset_coarseness=self.args.offset_coarseness,
                    separate_value_layer=self.args.separate_value_layer
                )
            else:
                network = LayerDecompositionAttentionMemoryDepthNet2D(
                    context_loader,
                    num_context_frames=args.num_context_frames,
                    in_channels=self.args.in_channels,
                    conv_channels=self.args.conv_channels,
                    valdim=self.args.valdim,
                    keydim=self.args.keydim,
                    topk=self.args.topk,
                    do_adjustment=True, 
                    max_frames=len(dataloader.dataset.frame_iterator),
                    br_coarseness=self.args.br_coarseness,
                    offset_coarseness=self.args.offset_coarseness,
                    separate_value_layer=self.args.separate_value_layer
                )
        elif self.args.model_type == "bottleneck_no_attention":
            if self.args.use_depth:
                raise NotImplementedError("Bottleneck model without context module is not supported with depth estimation")
            network = LayerDecompositionNet3DBottleneck(
                    context_loader=None,
                    num_context_frames=None,
                    in_channels=self.args.in_channels,
                    conv_channels=self.args.conv_channels,
                    do_adjustment=True, 
                    max_frames=len(dataloader.dataset.frame_iterator),
                    br_coarseness=self.args.br_coarseness,
                    offset_coarseness=self.args.offset_coarseness
            )
        elif self.args.model_type == "no_addons":
            if self.args.use_depth:
                raise NotImplementedError("Model without temporal add ons is not supported with depth estimation")
            network = Omnimatte(
                    in_channels=self.args.in_channels,
                    conv_channels=self.args.conv_channels,
                    do_adjustment=True, 
                    max_frames=len(dataloader.dataset.frame_iterator),
                    br_coarseness=self.args.br_coarseness,
                    offset_coarseness=self.args.offset_coarseness,
                    force_dynamics_layer=True
            )

        if self.args.device != "cpu":
            network = DataParallel(network).to(self.args.device)
        
        if self.args.continue_from != "":
            network.load_state_dict(torch.load(f"{self.args.continue_from}/reconstruction_weights.pth"))

        model = LayerDecompositer(
            dataloader, 
            context_loader,
            loss_module, 
            network, 
            writer,
            self.args.learning_rate, 
            results_root=self.args.out_dir, 
            batch_size=self.args.batch_size,
            n_epochs=self.args.n_epochs,
            save_freq=self.args.save_freq,
            separate_bg=not self.args.no_static_background,
            use_depth=self.args.use_depth,
            using_context=(self.args.model_type not in ["omnimatte", "no_addons", "bottleneck_no_attention"]),
            do_detail_transfer=self.args.do_detail_transfer
        )

        return model


if __name__ == "__main__":
    print("started")
    print(f"Running on {torch.cuda.device_count()} GPU{'s' if torch.cuda.device_count() > 1 else ''}")
    parser = ArgumentParser()
    parser.add_argument("--description", type=str, default="no description given", help="description of the experiment")

    dataset = "Videos"
    video = "nescio_1"
    directory_args = parser.add_argument_group("directories")
    directory_args.add_argument("--out_dir", type=str, default=f"results/omnimatte_test/{video}", 
        help="path to directory where results are saved")
    directory_args.add_argument("--initial_mask", nargs="+", default=[f"datasets/{dataset}/Annotations/{video}/00"],#, f"datasets/{dataset}/Annotations/{video}/01"], #, f"datasets/{dataset}/Annotations/{video}/02", f"datasets/{dataset}/Annotations/{video}/03"], 
        help="paths to the initial object masks or the directories containing the object masks")
    directory_args.add_argument("--img_dir", type=str, default=f"datasets/{dataset}/Images/{video}", 
        help="path to the directory in which the video frames are stored")
    directory_args.add_argument("--continue_from", type=str, default="", help="root directory of training run from which you wish to continue")

    model_args = parser.add_argument_group("model")
    model_args.add_argument("--model_type",        type=str, default="3d_bottleneck", choices=["3d_bottleneck", "fully_2d", "omnimatte", "bottleneck_no_attention", "no_addons"], help="The type of decomposition network to use")
    model_args.add_argument("--conv_channels",     type=int, default=16, help="base number of convolution channels in the convolutional neural networks")
    model_args.add_argument("--keydim",            type=int, default=8,  help="number of key channels in the attention memory network")
    model_args.add_argument("--valdim",            type=int, default=16, help="number of value channels in the attention memory network")
    model_args.add_argument("--in_channels",       type=int, default=16, help="number of channels in the input")
    model_args.add_argument("--br_coarseness",     type=int, default=10, help="Temporal coarseness of brightness adjustment")
    model_args.add_argument("--offset_coarseness", type=int, default=10, help="Temporal coarseness of background offset adjustment")
    model_args.add_argument("--topk",              type=int, default=0,  help="k value for topk channel selection in context distribution")
    model_args.add_argument("--use_2d_loss_module",   action="store_true", help="Use 2d loss module in stead of 3d loss module")
    model_args.add_argument("--no_static_background", action="store_true", help="Don't use separated static and dynamic background")
    model_args.add_argument("--use_depth",            action="store_true", help="specify that you want to use depth estimation as an input channel")
    model_args.add_argument("--corr_diff",            action="store_true", help="use corr_diff dynamics regularization in stead of alpha composite")
    model_args.add_argument("--alpha_reg_layers",     action="store_true", help="alpha regularization loss is l1 and l0 on alpha layers in stead of composite")
    model_args.add_argument("--use_alpha_detail_reg", action="store_true", help="use the alpha composite in detail bleed regularization")
    model_args.add_argument("--bottleneck_normal",    action="store_true", help="have a normal 3d conv as bottleneck in stead of a transposed conv")
    model_args.add_argument("--separate_value_layer", action="store_true", help="specify wether to use a separate value layer for the context and reconstruction encoder")
    model_args.add_argument("--do_detail_transfer",   action="store_true", help="specify whether to do detail transfer on the output at inference.")

    input_args = parser.add_argument_group("model input")
    input_args.add_argument("--composite_order",            type=str,   default="composite_order.txt", help="path to a text file containing the compositing order of the foreground objects")
    input_args.add_argument("--num_static_channels",        type=int,   default=5,    help="number of input channels that are static in time")
    input_args.add_argument("--timesteps",                  type=int,   default=4,    help="Temporal depth of the query input")
    input_args.add_argument("--num_context_frames",         type=int,   default=5,    help="period between frames that are added to memory")
    input_args.add_argument("--frame_height",               type=int,   default=256,  help="target height of the frames")
    input_args.add_argument("--frame_width",                type=int,   default=448,  help="target width of the frames")
    input_args.add_argument("--noise_temporal_coarseness",  type=int,   default=2,    help="temporal coarseness of the dynamic noise input")
    input_args.add_argument("--noise_upsample_size",        type=int,   default=16,   help="determines the spatial coarseness of both the spatial the and dynamic noise input")
    input_args.add_argument("--jitter_rate",                type=float, default=0.75, help="rate of applying jitter to the input")
    input_args.add_argument("--unsampled_dynamic_bg_input", action="store_true", help="specify whether the dynamic background layer will have uv sampled noise are not")

    training_param_args = parser.add_argument_group("training_parameters")
    training_param_args.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the reconstruction model")
    training_param_args.add_argument("--device",        type=str,   default="cpu", help="CUDA device")
    training_param_args.add_argument("--batch_size",    type=int,   default=1,     help="Batch size")
    training_param_args.add_argument("--n_epochs",      type=int,   default=1,     help="Number of epochs used for training")
    training_param_args.add_argument("--save_freq",     type=int,   default=5,     help="Frequency at which the intermediate results are saved")
    training_param_args.add_argument("--seed",          type=int,   default=1,     help="Random seed for libraries")
    training_param_args.add_argument("--n_gpus",        type=int,   default=torch.cuda.device_count(), help="Number of GPUs to use for training")

    # all lambda schedules should be a list starting with a staring value followed by pairs of UPDATE_EPOCH, NEW_VALUE
    lambdas = parser.add_argument_group("lambdas")
    lambdas.add_argument("--lambda_mask",              nargs="+", default=[50., 50, 0.], help="values for the lambda of the alpha_mask_bootstrap loss")
    lambdas.add_argument("--lambda_recon_flow",        nargs="+", default=[1.], help="lambda of the flow reconstruction loss")
    lambdas.add_argument("--lambda_recon_depth",       nargs="+", default=[1.], help="lambda of the depth reconstruction loss")
    lambdas.add_argument("--lambda_alpha_l0",          nargs="+", default=[0.005], help="lambda of the l0 part of the alpha regularization loss")
    lambdas.add_argument("--lambda_alpha_l1",          nargs="+", default=[0.01, 100, 0.], help="lambda of the l1 part of the alpha regularization loss")
    lambdas.add_argument("--lambda_stabilization",     nargs="+", default=[0.001], help="lambda of the camera stabilization loss")
    lambdas.add_argument("--lambda_detail_reg",        nargs="+", default=[10, 50, 0.1], help="lambda of the detail bleed regularization loss")
    lambdas.add_argument("--lambda_bg_scaling",        nargs="+", default=[1.], help="downscaling factor for dynamic background in alpha regularization.")

    lambdas.add_argument("--lambda_dynamics_reg_diff", nargs="+", default=[0.01], help="lambda of the difference part of the dynamics regularization loss")
    lambdas.add_argument("--lambda_dynamics_reg_corr", nargs="+", default=[0.005], help="lambda of the correlation part of the dynamics regularization loss")
    lambdas.add_argument("--lambda_dynamics_reg_l1",   nargs="+", default=[0.01], help="lambda of the difference part of the dynamics regularization loss")
    lambdas.add_argument("--lambda_dynamics_reg_l0",   nargs="+", default=[0.005], help="lambda of the correlation part of the dynamics regularization loss")

    lambdas.add_argument("--lambda_alpha_warp",        nargs="+", default=[0.005], help="lambda of the warped alpha estimation loss")
    lambdas.add_argument("--lambda_recon_warp",        nargs="+", default=[0.], help="lambda of the warped rgb reconstruction loss")

    pretrained_model_args = parser.add_argument_group("pretrained_models")
    pretrained_model_args.add_argument("--propagation_model", type=str, default="models/third_party/weights/topkstm.pth", 
        help="path to the weights of the mask propagation model")
    pretrained_model_args.add_argument("--flow_model", type=str, default="models/third_party/weights/raft.pth",
        help="path to the weights of the optical flow estimation model")
    pretrained_model_args.add_argument("--depth_model", type=str, default="models/third_party/weights/monodepth.pth",
        help="path to the weights of the depth estimation model")

    args = parser.parse_args()

    print("\n"+"#"*30+"\n")
    for arg in vars(args):
        print(arg, " "*(25 - len(arg)), getattr(args, arg))
    print("\n"+"#"*30+"\n")
    
    experiment_runner = ExperimentRunner(args)
    t0 = datetime.now()
    experiment_runner.start()
    print((datetime.now() - t0).total_seconds())

    print("done")