from .. import MaskPropVAE

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