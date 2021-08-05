from datetime import datetime
from torch.utils.data import DataLoader

from InputProcessing.inputProcessor import InputProcessor
from models.LayerDecomposition.layerDecomposition import LayerDecompositer
from models.LayerDecomposition.loss_functions import DecompositeLoss
from models.LayerDecomposition.modules import LayerDecompositionUNet

if __name__ == "__main__":
    video           = "tennis"
    results_dir     = f"results/layer_decomposition/{datetime.now()}"
    img_dir         = f"datasets/DAVIS_minisample/JPEGImages/480p/{video}"
    initial_mask    = f"datasets/DAVIS_minisample/Annotations/480p/{video}/00000.png"
    composite_order = f"datasets/DAVIS_minisample/Composite_order.txt"
    save_dir        = f"{results_dir}/decomposition"
    do_adjustment   = True
    batch_size      = 2

    input_processor = InputProcessor(img_dir, results_dir, initial_mask, composite_order, do_adjustment=do_adjustment)
    data_loader     = DataLoader(input_processor, batch_size=batch_size)
    loss_module     = DecompositeLoss()
    network         = LayerDecompositionUNet(do_adjustment=do_adjustment, max_frames=len(input_processor) + 1, coarseness=3)
    model           = LayerDecompositer(data_loader, loss_module, network, learning_rate=0.001, save_dir=save_dir, batch_size=batch_size)
    model.to("cuda")

    model.train(2000)