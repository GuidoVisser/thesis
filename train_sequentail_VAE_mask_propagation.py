# general imports
import argparse
import numpy as np
import torch

# Model trainer imports
from utils.modelTrainer import ModelTrainer
from utils.utils import seed_all

# model specific imports
from datasets import DAVISSequenceDataset
from utils.initialize_models import initialize_RAFT
from utils.transforms import get_transforms
from models.SequentialMaskPropVAE import initialize_SeqMaskPropVAE, train_vae, test_vae, demo_model


def main(args):
    """
    Main Function for the full training & evaluation loop of a VAE model.
    Make use of a separate train function and a test function for both
    validation and testing (testing only once after training).
    Inputs:
        args - Namespace object from the argument parser
    """
    if args.seed is not None:
        seed_all(args.seed)

    # create dataset
    dataset = DAVISSequenceDataset(args.data_dir, get_transforms(train=True))

    # Create models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = initialize_SeqMaskPropVAE(args, device)
    RAFT_model = initialize_RAFT(args, device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainer = ModelTrainer(args, \
                           model,
                           [RAFT_model], \
                           dataset, \
                           optimizer, \
                           train_vae, \
                           test_vae, \
                           demo_model)
    
    trainer.train()
    
    return


if __name__ == '__main__':
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
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=100, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')

    # directories
    parser.add_argument('--data_dir', default='datasets/DAVIS_sample_tennis/', type=str,
                        help='Directory where data is stored')
    parser.add_argument('--log_dir', default='results/DAVIS_sample_tennis_maskpropvae', type=str,
                        help='Directory where the PyTorch logs should be created.')

    # RAFT 
    parser.add_argument('--RAFT_weights', default='models/weights/zip_serialization_false/raft-things.pth', 
                        help='restore a RAFT checkpoint')
    parser.add_argument('--small', action='store_true', 
                        help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', 
                        help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', 
                        help='use efficent correlation implementation')
    
    parser.add_argument("--demo_freq", default=5, type=int,
                        help="Frequency at which the demo fucntion is called")


    args = parser.parse_args()

    main(args)
