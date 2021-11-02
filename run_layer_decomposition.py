from argparse import ArgumentParser
from datetime import datetime
from numpy.random import shuffle
from torch.utils.data import DataLoader
import torch
import numpy as np
from os import path
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.tensorboard import SummaryWriter

from InputProcessing.inputProcessor import InputProcessor
from models.DynamicLayerDecomposition.attention_memory_modules import AttentionMemoryNetwork, MemoryReader
from models.DynamicLayerDecomposition.layerDecomposition import LayerDecompositer
from models.DynamicLayerDecomposition.loss_functions import DecompositeLoss
from models.DynamicLayerDecomposition.modules import LayerDecompositionUNet

from utils.distributed_training import setup, cleanup, spawn_multiprocessor
from utils.demo import create_decomposite_demo
from utils.utils import create_dir
from models.DynamicLayerDecomposition.model_config import CONFIG, update_config, save_config 

def distributed_training(rank, n_gpus, model):
    setup(rank, n_gpus)

    model.net = model.net.to(rank)                
    model.net = DistributedDataParallel(model.net, device_ids=[rank])

    model.train(rank)

    cleanup()


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    writer = SummaryWriter(args.out_dir)

    input_processor = InputProcessor(
        args.img_dir, 
        args.out_dir, 
        args.initial_mask, 
        args.composite_order, 
        do_jitter = True, 
        propagation_model = args.propagation_model, 
        flow_model = args.flow_model
    )

    data_loader = DataLoader(
        input_processor, 
        batch_size=args.batch_size,
        shuffle=True
    )

    loss_module = DecompositeLoss(
        args.alpha_bootstr_thresh,
        args.lambda_mask,
        args.lambda_recon_flow,
        args.lambda_recon_warp,
        args.lambda_alpha_warp,
        args.lambda_alpha_l0,
        args.lambda_alpha_l1,
        args.lambda_stabilization
    )

    if isinstance(args.initial_mask, str):
        num_mem_nets = 2
    else:
        raise ValueError("TODO: Make sure the number of objects is correctly passed to the memory network")
    
    # use 1 memory network for all layers
    if args.experiment_config not in [1, 4]:
        num_mem_nets = 1

    attention_memory = DataParallel(AttentionMemoryNetwork(
        args.keydim,
        args.valdim,
        num_mem_nets,
        args.mem_freq,
        input_processor,
        args.propagation_model
    )).to(args.device)

    memory_reader = MemoryReader(
        args.keydim,
        args.valdim,
        num_mem_nets,
        args.propagation_model,
        args.experiment_config
    )

    network = DataParallel(LayerDecompositionUNet(
        memory_reader,
        do_adjustment=True, 
        max_frames=len(input_processor) + 1, # +1 because len(input_processor) specifies the number of PAIRS of frames
        coarseness=args.coarseness,
        experiment_config=args.experiment_config
    )).to(args.device)

    model = LayerDecompositer(
        data_loader, 
        loss_module, 
        network, 
        attention_memory,
        writer,
        args.learning_rate, 
        args.memory_learning_rate, 
        results_root=args.out_dir, 
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        save_freq=args.save_freq
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

    dataset = "Jaap_Jelle"
    video = "kruispunt_rijks"
    directory_args = parser.add_argument_group("directories")
    directory_args.add_argument("--out_dir", type=str, default=f"results/layer_decomposition_dynamic/{video}", 
        help="path to directory where results are saved")
    directory_args.add_argument("--initial_mask", type=str, default=f"datasets/{dataset}/Annotations/{video}/combined/00006.png", 
        help="path to the initial mask")
    directory_args.add_argument("--img_dir", type=str, default=f"datasets/{dataset}/JPEGImages/480p/{video}", 
        help="path to the directory in which the video frames are stored")
    
    reconstruction_model_args = parser.add_argument_group("reconstruction_model")
    reconstruction_model_args.add_argument("--composite_order", type=str, 
        help="path to a text file containing the compositing order of the foreground objects")
    reconstruction_model_args.add_argument("--coarseness", type=int, default=10, 
        help="Temporal coarseness of camera adjustment parameters")

    memory_network_args = parser.add_argument_group("memory_network")
    memory_network_args.add_argument("--keydim", type=int, default=128, help="number of key channels in the attention memory network")
    memory_network_args.add_argument("--valdim", type=int, default=512, help="number of value channels in the attention memory network")
    memory_network_args.add_argument("--mem_freq", type=int, default=9, help="specifies the interval between the frames that are added to the memory network")

    training_param_args = parser.add_argument_group("training_parameters")
    training_param_args.add_argument("--batch_size", type=int, default=1, help="Batch size")
    training_param_args.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the reconstruction model")
    training_param_args.add_argument("--memory_learning_rate", type=float, default=0.001, help="Learning rate for the memory encoder")
    training_param_args.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    training_param_args.add_argument("--n_epochs", type=int, default=1, help="Number of epochs used for training")
    training_param_args.add_argument("--save_freq", type=int, default=30, help="Frequency at which the intermediate results are saved")
    training_param_args.add_argument("--n_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for training")
    training_param_args.add_argument("--seed", type=int, default=1, help="Random seed for libraries")
    training_param_args.add_argument("--alpha_bootstr_thresh", type=float, default=5e-3, help="Threshold for the alpha bootstrap loss. If the loss comes under this the lambda is decreased")
    training_param_args.add_argument("--experiment_config", type=int, default=2, help="configuration id for the experiment that is being run")

    lambdas = parser.add_argument_group("lambdas")
    lambdas.add_argument("--lambda_mask", type=float, default=50., help="starting value for the lambda of the alpha_mask_bootstrap loss")
    lambdas.add_argument("--lambda_recon_flow", type=float, default=1., help="lambda of the flow reconstruction loss")
    lambdas.add_argument("--lambda_recon_warp", type=float, default=0., help="lambda of the warped rgb reconstruction loss")
    lambdas.add_argument("--lambda_alpha_warp", type=float, default=0.005, help="lambda of the warped alpha estimation loss")
    lambdas.add_argument("--lambda_alpha_l0", type=float, default=0.005, help="lambda of the l0 part of the alpha regularization loss")
    lambdas.add_argument("--lambda_alpha_l1", type=float, default=0.01, help="lambda of the l1 part of the alpha regularization loss")
    lambdas.add_argument("--lambda_stabilization", type=float, default=0.001, help="lambda of the camera stabilization loss")

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