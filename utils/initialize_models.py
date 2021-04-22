import torch

from models.RAFT import RAFT
from models.MaskPropagationVAE import MaskPropVAE


def initialize_RAFT(args, device, eval=True):
    """
    Initialize a RAFT model

    Args:
        args (namespace): namespace containing configuration details on the RAFT model. Should contain:
                            args.RAFT_weights       (str) : directory to the weights of the RAFT model
                            args.small              (bool): use the small RAFT model
                            args.mixed_precision    (bool): use mixed precision
                            args.alternate_corr     (bool): use efficient correlation calculation
        device (str): denotes what cuda device should be used.
        eval (bool): use evaluation mode
    
    Returns:
        model (nn.Module): Fully initialized RAFT model
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.RAFT_weights))
    model = model.module
    model.to(device)
    if eval:
        model.eval()

    return model


def initialize_MaskPropVAE(args, device, eval=False):
    """
    Initialize a Mask Propagation VAE model

    Args:
        args (namespace): namespace containing configuration details on the MaskPropVAE model. Should contain:
                            args.num_filters    (int): Number of filters to be used in the convolutional layers
                            args.z_dim          (int): Number of dimensions in the latent space.
        device (str): denotes what cuda device should be used.
        eval (bool): use evaluation mode
    
    Returns:
        model (nn.Module): Fully initialized MaskPropVAE model
    """
    model = MaskPropVAE(num_filters=args.num_filters, z_dim=args.z_dim)
    model = model.to(device)
    if eval:
        model.eval()

    return model