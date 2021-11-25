"""Script for training an Omnimatte model on a video.

You need to specify the dataset ('--dataroot') and experiment name ('--name').

Example:
    python train.py --dataroot ./datasets/tennis --name tennis --gpu_ids 0,1

The script first creates a model, dataset, and visualizer given the options.
It then does standard network training. During training, it also visualizes/saves the images, prints/saves the loss
plot, and saves the model.
Use '--continue_train' to resume your previous training.

See options/base_options.py and options/train_options.py for more training options.
"""
import time

from models.third_party.Omnimatte.options.train_options import TrainOptions
from models.third_party.Omnimatte.third_party.data import create_dataset
from models.third_party.Omnimatte.third_party.models import create_model
from models.third_party.Omnimatte.third_party.util.visualizer import Visualizer
import torch
import numpy as np

from argparse import ArgumentParser


def main(args):
    trainopt = TrainOptions()
    opt = trainopt.parse()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    opt.n_epochs       = int(opt.n_steps / np.ceil(dataset_size / opt.batch_size))
    opt.n_epochs_decay = int(opt.n_steps_decay / np.ceil(dataset_size / opt.batch_size))

    model = create_model(opt)
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)

    train(model, dataset, visualizer, opt)


def train(model, dataset, visualizer, opt):
    dataset_size = len(dataset)
    total_iters = 0  # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time   = time.time()  # timer for data loading per iteration
        
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_lambdas(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch

            iter_start_time = time.time()   # timer for computation per iteration

            if i % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter  += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if i % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()

        if epoch % opt.display_freq == 1:   # display images on visdom and save images to a HTML file
            save_result = epoch % opt.update_html_freq == 1
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if epoch % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> epochs
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'epoch_%d' % epoch if opt.save_by_epoch else 'latest'
            model.save_networks(save_suffix)

        model.update_learning_rate()    # update learning rates at the end of every epoch.
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


if __name__ == '__main__':
    print("started")
    print(f"Running on {torch.cuda.device_count()} GPU{'s' if torch.cuda.device_count() > 1 else ''}")
    parser = ArgumentParser()

    video = "tennis"
    parser.add_argument("--out_dir", type=str, default=f"results/layer_decomposition/test_new", 
        help="path to directory where results are saved")
    parser.add_argument("--initial_mask", type=str, default=f"datasets/Omnimatte_tennis/mask/01/0001.png", 
        help="path to the initial mask")
    parser.add_argument("--img_dir", type=str, default=f"datasets/Omnimatte_tennis/rgb", 
        help="path to the directory in which the video frames are stored")
    parser.add_argument("--composite_order", type=str, 
        help="path to a text file containing the compositing order of the foreground objects")

    parser.add_argument("--do_adjustment", dest="do_adjustment", action='store_true', help="Specifies whether to use learnable camera adjustment")
    parser.add_argument("--dont_adjustment", dest="do_adjustment", action='store_false', help="Specifies whether to use learnable camera adjustment")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate during training")
    parser.add_argument("--coarseness", type=int, default=10, help="Temporal coarseness of camera adjustment parameters")
    parser.add_argument("--device", type=str, default="cuda", help="CUDA device")
    parser.add_argument("--n_epochs", type=int, default=176, help="Number of epochs used for training")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency at which the intermediate results are saved")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for libraries")

    parser.add_argument("--propagation_model", type=str, default="models/third_party/weights/propagation_model.pth", 
        help="path to the weights of the mask propagation model")
    parser.add_argument("--flow_model", type=str, default="models/third_party/weights/raft-things.pth",
        help="path to the optical flow estimation model")

    args = parser.parse_args()
    print(args.do_adjustment)
    main(args)
    print("done")

