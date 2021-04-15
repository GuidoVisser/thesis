import argparse
import os
import datetime
import random

from tqdm import tqdm, trange
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import torch.utils.data as data

from MaskPropagationVAE import VAE
from transforms import get_transforms
from datasets import DAVISPairsDataset

from FGVC.RAFT import RAFT
from FGVC.RAFT import utils as RAFT_utils
from FGVC.RAFT.utils.utils import InputPadder


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_train_and_valid_loaders(args, train=True):
    """
    Create a data loader object

    Inputs:
        batch_size (int): batch size for training
        num_workers (int): number of workers on the GPU
    """
    dataset = DAVISPairsDataset(args.data_dir, args.mask_dir, get_transforms(train=train))

    train_valid_split = int(0.8 * len(dataset))

    train_set = data.dataset.Subset(dataset, np.arange(train_valid_split))
    valid_set = data.dataset.Subset(dataset, np.arange(train_valid_split, len(dataset)))

    train_loader = data.DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    valid_loader = data.DataLoader(
        valid_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    return train_loader, valid_loader


def calculate_batch_flow(flow_model, source_batch, target_batch):

    _, flow = flow_model(source_batch, target_batch, iters=1, test_mode=True)

    return flow

def sample_and_save(model, epoch, summary_writer, batch_size=64):
    """
    Function that generates and saves samples from the VAE.  The generated
    samples and mean images should be saved, and can eventually be added to a
    TensorBoard logger if wanted.
    Inputs:
        model - The VAE model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        batch_size - Number of images to generate/sample
    """

    sample_batch = model.sample(batch_size)
    samples = sample_batch[0]
    sample_means = sample_batch[1]

    grid = make_grid(samples)
    grid_of_means = make_grid(sample_means)
    save_image(grid, f"{summary_writer.log_dir}/sample_epoch_{epoch}.png")
    save_image(grid_of_means, f"{summary_writer.log_dir}/means_epoch_{epoch}.png")


@torch.no_grad()
def test_vae(model, flow_model, data_loader):
    """
    Function for testing a model on a dataset.
    Inputs:
        model - VAE model to test
        data_loader - Data Loader for the dataset you want to test on.
    Outputs:
        average_bpd - Average BPD
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """

    # initialize padder object for RAFT
    sample_frame, _, _, _ = next(iter(data_loader))
    raft_padder = InputPadder(sample_frame.size())
    del sample_frame

    average_bpd = 0
    average_rec_loss = 0
    average_reg_loss = 0

    for step, batch in enumerate(data_loader):
        
        # prepare batch
        source_frames, source_masks, target_frames, target_masks = [input.to('cuda') for input in batch]
        source_frames, source_masks, target_frames, target_masks = raft_padder.pad(source_frames, 
                                                                                   source_masks, 
                                                                                   target_frames, 
                                                                                   target_masks)
        # calculating flow
        flow = calculate_batch_flow(flow_model, source_frames, target_frames)

        # calculating forward pass
        L_rec, L_reg, bpd = model.forward(source_masks, flow, source_frames, target_masks)
        
        # keep a running average
        current = step + 1
        prev_weight = step / current
        average_bpd = average_bpd * prev_weight + bpd / current
        average_rec_loss = average_rec_loss * prev_weight + L_rec / current
        average_reg_loss = average_reg_loss * prev_weight + L_reg / current  

    return average_bpd, average_rec_loss, average_reg_loss


def train_vae(model, flow_model, train_loader, optimizer):
    """
    Function for training a model on a dataset. Train the model for one epoch.
    Inputs:
        model - VAE model to train
        train_loader - Data Loader for the dataset you want to train on
        optimizer - The optimizer used to update the parameters
    Outputs:
        average_bpd - Average BPD
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """

    # initialize padder object for RAFT
    sample_frame, _, _, _ = next(iter(train_loader))
    raft_padder = InputPadder(sample_frame.size())
    del sample_frame

    average_bpd = 0
    average_rec_loss = 0
    average_reg_loss = 0

    for step, batch in enumerate(train_loader):
        print("<=== getting batch  ===>")
        source_frames, source_masks, target_frames, target_masks = [input.to(model.device) for input in batch]
        source_frames, source_masks, target_frames, target_masks = raft_padder.pad(source_frames, 
                                                                                   source_masks, 
                                                                                   target_frames, 
                                                                                   target_masks)

        print("<=== calculating flow ===>")
        flow = calculate_batch_flow(flow_model, source_frames, target_frames)

        print("<=== model forward  ===>")
        L_rec, L_reg, bpd = model.forward(source_masks, flow, source_frames, target_masks)

        print("<=== model backward ===>\n")
        # update model
        model.zero_grad()
        bpd.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        # DEBUG
        # for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        #     print(p.grad.data.norm(2).item())

        # keep a running average
        current = step + 1
        prev_weight = step / current
        average_bpd = average_bpd * prev_weight + bpd / current
        average_rec_loss = average_rec_loss * prev_weight + L_rec / current
        average_reg_loss = average_reg_loss * prev_weight + L_reg / current  

    return average_bpd, average_rec_loss, average_reg_loss


def main(args):
    """
    Main Function for the full training & evaluation loop of a VAE model.
    Make use of a separate train function and a test function for both
    validation and testing (testing only once after training).
    Inputs:
        args - Namespace object from the argument parser
    """
    if args.seed is not None:
        seed_everything(args.seed)

    # Prepare logging
    experiment_dir = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpoint_dir = os.path.join(
        experiment_dir, 'checkpoints')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_writer = SummaryWriter(experiment_dir)

    # Load dataset
    train_loader, val_loader = create_train_and_valid_loaders(args)

    # Create model
    model = VAE(num_filters=args.num_filters, z_dim=args.z_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    RAFT_model = torch.nn.DataParallel(RAFT(args))
    RAFT_model.load_state_dict(torch.load(args.flow_model))
    RAFT_model = RAFT_model.module
    RAFT_model.to(device)
    RAFT_model.eval()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # # # Sample image grid before training starts
    # sample_and_save(model, 0, summary_writer, 64)

    # Tracking variables for finding best model
    best_val_bpd = float('inf')
    best_epoch_idx = 0
    print(f"Using device {device}")
    epoch_iterator = (trange(1, args.epochs + 1, desc=f"VAE")
                      if args.progress_bar else range(1, args.epochs + 1))
    for epoch in epoch_iterator:
        if not args.progress_bar:
            print(f"Epoch : {epoch}")
        # Training epoch
        train_iterator = (tqdm(train_loader, desc="Training", leave=False)
                          if args.progress_bar else train_loader)
        epoch_train_bpd, train_rec_loss, train_reg_loss = train_vae(
            model, RAFT_model, train_iterator, optimizer)

        # Validation epoch
        val_iterator = (tqdm(val_loader, desc="Testing", leave=False)
                        if args.progress_bar else val_loader)
        epoch_val_bpd, val_rec_loss, val_reg_loss = test_vae(model, RAFT_model, val_iterator)

        # Logging to TensorBoard
        summary_writer.add_scalars(
            "BPD", {"train": epoch_train_bpd, "val": epoch_val_bpd}, epoch)
        summary_writer.add_scalars(
            "Reconstruction Loss", {"train": train_rec_loss, "val": val_rec_loss}, epoch)
        summary_writer.add_scalars(
            "Regularization Loss", {"train": train_reg_loss, "val": train_reg_loss}, epoch)
        summary_writer.add_scalars(
            "Negative ELBO", {"train": train_rec_loss + train_reg_loss, "val": val_rec_loss + val_reg_loss}, epoch)

        # if epoch % 5 == 0:
        #     sample_and_save(model, epoch, summary_writer, 64)

        # Saving best model
        if epoch_val_bpd < best_val_bpd:
            best_val_bpd = epoch_val_bpd
            best_epoch_idx = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "epoch.pt"))

    # Load best model for test
    print(f"Best epoch: {best_epoch_idx}. Load model for testing.")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "epoch.pt")))

    # Test epoch
    test_loader = (tqdm(test_loader, desc="Testing", leave=False)
                   if args.progress_bar else test_loader)
    test_bpd, _, _ = test_vae(model, test_loader)
    print(f"Test BPD: {test_bpd}")
    summary_writer.add_scalars("BPD", {"test": test_bpd}, best_epoch_idx)

    # # Manifold generation
    # if args.z_dim == 2:
    #     img_grid = visualize_manifold(model.decoder)
    #     save_image(img_grid, os.path.join(experiment_dir, 'vae_manifold.png'),
    #                normalize=False)

    # return test_bpd
    return


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--z_dim', default=200, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--num_filters', default=32, type=int,
                        help='Number of channels/filters to use in the CNN encoder/decoder.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--data_dir', default='datasets/DAVIS/JPEGImages/480p/', type=str,
                        help='Directory where data is stored')
    parser.add_argument('--mask_dir', default='datasets/DAVIS/Annotations/480p/', type=str,
                        help='Directory where masks are stored')
    parser.add_argument('--log_dir', default='results/VAE_logs', type=str,
                        help='Directory where the PyTorch logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    # RAFT 
    parser.add_argument('--flow_model', default='FGVC/weight/zip_serialization_false/raft-things.pth', 
                        help='restore a RAFT checkpoint')
    parser.add_argument('--small', action='store_true', 
                        help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', 
                        help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', 
                        help='use efficent correlation implementation')
    
    args = parser.parse_args()

    main(args)
