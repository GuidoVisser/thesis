from argparse import ArgumentParser
from datetime import datetime
from torch.utils.data import DataLoader

from InputProcessing.inputProcessor import InputProcessor
from models.LayerDecomposition.layerDecomposition import LayerDecompositer
from models.LayerDecomposition.loss_functions import DecompositeLoss
from models.LayerDecomposition.modules import LayerDecompositionUNet

def main(args):

    input_processor = InputProcessor(
        args.img_dir, 
        args.out_dir, 
        args.initial_mask, 
        args.composite_order, 
        do_adjustment=args.do_adjustment, 
        propagation_model=args.propagation_model, 
        flow_model=args.flow_model
    )

    data_loader = DataLoader(
        input_processor, 
        batch_size=args.batch_size
    )
    
    loss_module = DecompositeLoss()

    network = LayerDecompositionUNet(
        do_adjustment=args.do_adjustment, 
        max_frames=len(input_processor) + 1, 
        coarseness=args.coarseness
    )

    model = LayerDecompositer(
        data_loader, 
        loss_module, 
        network, 
        args.learning_rate, 
        results_root=args.out_dir, 
        batch_size=args.batch_size
    )

    model.to(args.device)

    model.train(args.n_epochs)

if __name__ == "__main__":
    parser = ArgumentParser()

    video = "tennis"
    parser.add_argument("--out_dir", type=str, default=f"results/layer_decomposition/{datetime.now()}", 
        help="path to directory where results are saved")
    parser.add_argument("--initial_mask", type=str, default=f"datasets/DAVIS_minisample/Annotations/480p/{video}/00000.png", 
        help="path to the initial mask")
    parser.add_argument("--img_dir", type=str, default=f"datasets/DAVIS_minisample/JPEGImages/480p/{video}", 
        help="path to the directory in which the video frames are stored")
    parser.add_argument("--composite_order", type=str, 
        help="path to a text file containing the compositing order of the foreground objects")

    parser.add_argument("--do_adjustment", type=bool, default=True, help="Specifies whether to use learnable camera adjustment")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate during training")
    parser.add_argument("--coarseness", type=int, default=10, help="Temporal coarseness of camera adjustment parameters")
    parser.add_argument("--device", type=str, default="cuda", help="CUDA device")
    parser.add_argument("--n_epochs", type=int, default=2000, help="Number of epochs used for training")

    parser.add_argument("--propagation_model", type=str, default="models/third_party/weights/propagation_model.pth", 
        help="path to the weights of the mask propagation model")
    parser.add_argument("--flow_model", type=str, default="models/third_party/weights/raft-things.pth",
        help="path to the optical flow estimation model")