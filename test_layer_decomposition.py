from datetime import datetime
from torch.utils.data import DataLoader

from utils.utils import create_dirs
from InputProcessing.inputProcessor import InputProcessor
from models.LayerDecomposition.layerDecomposition import LayerDecompositer
from models.LayerDecomposition.loss_functions import DecompositeLoss
from models.LayerDecomposition.modules import LayerDecompositionUNet

if __name__ == "__main__":
    video           = "tennis"
    results_dir     = "results/layer_decomposition"
    T               = datetime.now()

    img_dir         = f"datasets/DAVIS_minisample/JPEGImages/480p/{video}"
    initial_mask    = f"datasets/DAVIS_minisample/Annotations/480p/{video}/00000.png"
    mask_dir        = f"{results_dir}/{T}/masks/{video}"
    flow_dir        = f"{results_dir}/{T}/flow/{video}"
    background_dir  = f"{results_dir}/{T}/background/{video}"
    demo_dir        = f"{results_dir}/{T}"
    create_dirs(mask_dir, flow_dir, background_dir)
    
    ip = InputProcessor(img_dir, mask_dir, initial_mask, flow_dir, background_dir)
    data_loader = DataLoader(ip)

    loss_module = DecompositeLoss()
    network = LayerDecompositionUNet()
    model = LayerDecompositer(data_loader, loss_module, network, learning_rate=0.1)

    model.train()