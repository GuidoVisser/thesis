from argparse import ArgumentParser
from datetime import datetime
from numpy.random import shuffle
from torch.utils.data import DataLoader
import torch
import numpy as np
from os import path
from torch.nn.parallel import DistributedDataParallel, DataParallel

from InputProcessing.inputProcessor import InputProcessor
from models.DynamicLayerDecomposition.attention_memory_modules import AttentionMemoryNetwork, MemoryReader
from models.DynamicLayerDecomposition.layerDecomposition import LayerDecompositer
from models.DynamicLayerDecomposition.loss_functions import DecompositeLoss
from models.DynamicLayerDecomposition.modules import LayerDecompositionUNet

from utils.distributed_training import setup, cleanup, spawn_multiprocessor
from utils.demo import create_decomposite_demo

def distributed_training(rank, n_gpus, model):
    setup(rank, n_gpus)

    model.net = model.net.to(rank)                
    model.net = DistributedDataParallel(model.net, device_ids=[rank])

    model.train(rank)

    cleanup()


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    loss_module = DecompositeLoss()

    if isinstance(args.initial_mask, str):
        num_objects = 2
    else:
        raise ValueError("TODO: Make sure the number of objects is correctly passed to the memory network")
    
    attention_memory = AttentionMemoryNetwork(
        args.keydim,
        args.valdim,
        num_objects,
        args.mem_freq,
        input_processor.frame_iterator,
    ).to(args.mem_device)

    memory_reader = MemoryReader(
        args.keydim,
        args.valdim,
        num_objects
    )

    network = DataParallel(LayerDecompositionUNet(
        memory_reader,
        do_adjustment=True, 
        max_frames=len(input_processor) + 1, # +1 because len(input_processor) specifies the number of PAIRS of frames
        coarseness=args.coarseness
    )).to(args.device)

    model = LayerDecompositer(
        data_loader, 
        loss_module, 
        network, 
        attention_memory,
        args.learning_rate, 
        results_root=args.out_dir, 
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        save_freq=args.save_freq
    )

    model.train(args.device)

    # # Set up for inference
    # input_processor.do_jitter = False
    # data_loader.shuffle = False
    # network.load_state_dict(torch.load(path.join(args.out_dir, "weights.pth")))

    # model.decomposite(args.device)

    create_decomposite_demo(path.join(args.out_dir, "decomposition/inference"))

if __name__ == "__main__":
    print("started")
    print(f"Running on {torch.cuda.device_count()} GPU{'s' if torch.cuda.device_count() > 1 else ''}")
    parser = ArgumentParser()

    video = "tennis"
    parser.add_argument("--out_dir", type=str, default=f"results/layer_decomposition_dynamic/{video}", 
        help="path to directory where results are saved")
    parser.add_argument("--initial_mask", type=str, default=f"datasets/DAVIS/Annotations/480p/{video}/00000.png", 
        help="path to the initial mask")
    parser.add_argument("--img_dir", type=str, default=f"datasets/DAVIS/JPEGImages/480p/{video}", 
        help="path to the directory in which the video frames are stored")
    parser.add_argument("--composite_order", type=str, 
        help="path to a text file containing the compositing order of the foreground objects")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate during training")
    parser.add_argument("--coarseness", type=int, default=10, help="Temporal coarseness of camera adjustment parameters")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs used for training")
    parser.add_argument("--save_freq", type=int, default=30, help="Frequency at which the intermediate results are saved")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for libraries")

    parser.add_argument("--keydim", type=int, default=128, help="number of key channels in the attention memory network")
    parser.add_argument("--valdim", type=int, default=512, help="number of value channels in the attention memory network")
    parser.add_argument("--mem_freq", type=int, default=30, help="specifies the interval between the frames that are added to the memory network")
    parser.add_argument("--mem_device", type=str, default="cuda:0", help="specifies the device on which the memory network lives")

    parser.add_argument("--propagation_model", type=str, default="models/third_party/weights/propagation_model.pth", 
        help="path to the weights of the mask propagation model")
    parser.add_argument("--flow_model", type=str, default="models/third_party/weights/raft-things.pth",
        help="path to the optical flow estimation model")

    args = parser.parse_args()
    main(args)
    print("done")