from .. import UNet

def initialize_unet(in_channels, out_channels, device, eval=False):
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
    model = UNet(in_channels, out_channels)
    model = model.to(device)
    if eval:
        model.eval()

    return model