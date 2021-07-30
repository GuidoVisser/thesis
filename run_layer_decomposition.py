from datetime import datetime
from torch.utils.data import DataLoader

from InputProcessing.inputProcessor import InputProcessor
from models.LayerDecomposition.layerDecomposition import LayerDecompositer
from models.LayerDecomposition.loss_functions import DecompositeLoss
from models.LayerDecomposition.modules import LayerDecompositionUNet

if __name__ == "__main__":
    video        = "tennis"
    results_dir  = f"results/layer_decomposition/{datetime.now()}"
    img_dir      = f"datasets/DAVIS_minisample/JPEGImages/480p/{video}"
    initial_mask = f"datasets/DAVIS_minisample/Annotations/480p/{video}/00000.png"
    save_dir     = f"{results_dir}/decomposition"

    input_processor = InputProcessor(img_dir, results_dir, initial_mask)
    data_loader     = DataLoader(input_processor)
    loss_module     = DecompositeLoss()
    network         = LayerDecompositionUNet()
    model           = LayerDecompositer(data_loader, loss_module, network, learning_rate=0.1, save_dir=save_dir)
    model.to("cuda")

    model.train(2000)