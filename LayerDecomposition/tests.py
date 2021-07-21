import torch
from LayerDecomposition.compositing import composite

def test_composite_shape(in_shape, out_shape, order):

    input = torch.rand(in_shape)
    output = composite(input, order)

    assert output.shape == out_shape, f"Output shape incorrect: Expected output to have shape {out_shape} but got {output.shape}."

    return 1
    