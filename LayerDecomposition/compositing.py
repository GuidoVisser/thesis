import torch

def composite(layers: torch.Tensor, layer_order: list) -> torch.Tensor:
    """
    Composite several layers together into a single image tensor

    Dimensionality:
        This function maps a N RGBa layers into a single RGB image
        [T x N x C x H x W] -> [T x C-1 x H x W]

    Args:
        layers (torch.Tensor): Tensor containing the image layers
        layer_order (list): order in which the layers should be composited
    """
    for i, layer_id in enumerate(layer_order):

        if i == 0:
            composite = layers[:, layer_id, :3]
            continue

        alpha = layers[:, layer_id, 3:4] * .5 + .5
        composite = layers[:, layer_id, :3] * alpha + composite * (1. - alpha)

    return composite