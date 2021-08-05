import cv2
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from os import path

from utils.utils import create_dirs
from models.third_party.RAFT.utils.flow_viz import flow_to_image


class LayerDecompositer(nn.Module):
    def __init__(self,
                 dataloader: DataLoader,
                 loss_module: nn.Module,
                 network: nn.Module,
                 learning_rate: float,
                 results_root: str,
                 batch_size: int):
        super().__init__()

        self.dataloader = dataloader
        self.loss_module = loss_module
        self.net = network
        self.optimizer = Adam(self.net.parameters(), learning_rate)

        self.save_dir = f"{results_root}/decomposition"
        self.save_freq = 10
        self.batch_size = batch_size

    def train(self, n_epochs):

        for epoch in range(n_epochs):
            if epoch % self.save_freq == 0:
                create_dirs(path.join(self.save_dir, f"{epoch:03}/background"), 
                            path.join(self.save_dir, f"{epoch:03}/foreground"),
                            path.join(self.save_dir, f"{epoch:03}/alpha"),
                            path.join(self.save_dir, f"{epoch:03}/reconstruction"),
                            path.join(self.save_dir, f"{epoch:03}/ground_truth"),
                            path.join(self.save_dir, f"{epoch:03}/flow"))
            print(f"Epoch: {epoch} / {n_epochs}")

            for i, (input, targets) in enumerate(self.dataloader):

                output = self.net(input)
                loss = self.loss_module(output, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if epoch % self.save_freq == 0:
                    if i == 0:
                        self.visualize_and_save_output(output, targets, i, epoch, 0)    
                    self.visualize_and_save_output(output, targets, i, epoch, 1)

        torch.save(self.net.state_dict(), path.join(self.save_dir, "weights.pth"))

    def visualize_and_save_output(self, model_output, targets, i, epoch, t):
        """
        Save the output of the model 
        """

        rgba_layers    = model_output["layers_rgba"]
        flow_layers    = model_output["layers_flow"]
        reconstruction = model_output["rgba_reconstruction"]

        gt_rgb = targets["rgb"]

        # NOTE: current_batch_size is not necessarily the same as self.batch_size at the end of an epoch
        current_batch_size, _, n_layers, _, _, _ = rgba_layers.shape 

        for b in range(current_batch_size):

            # background
            background_rgb     = torch.clone(rgba_layers[b, t, 0, :3]).detach()
            reconstruction_rgb = torch.clone(reconstruction[b, t, :3]).detach()
            gt_rgb_batch       = gt_rgb[b, t]

            background_img     = (background_rgb.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255
            reconstruction_img = (reconstruction_rgb.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255
            gt_rgb_img         = (gt_rgb_batch.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2. * 255


            img_name = f"{(i + t)*self.batch_size + b:05}.png"
            cv2.imwrite(path.join(self.save_dir, f"{epoch:03}/background/{img_name}"), background_img)
            cv2.imwrite(path.join(self.save_dir, f"{epoch:03}/reconstruction/{img_name}"), reconstruction_img)
            cv2.imwrite(path.join(self.save_dir, f"{epoch:03}/ground_truth/{img_name}"), gt_rgb_img)

            for l in range(1, n_layers):
                create_dirs(path.join(self.save_dir, f"{epoch:03}/foreground/{l:02}"),
                            path.join(self.save_dir, f"{epoch:03}/alpha/{l:02}"),
                            path.join(self.save_dir, f"{epoch:03}/flow/{l:02}"))
                foreground_rgb     = torch.clone(rgba_layers[b, t, l, :3]).detach()
                foreground_flow    = torch.clone(flow_layers[b, t, l]).detach()
                foreground_alpha   = torch.clone(rgba_layers[b, t, l, 3]).detach()

                foreground_img      = (foreground_rgb.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255
                alpha_img           = (foreground_alpha.cpu().numpy() + 1) / 2. * 255
                foreground_flow_img = flow_to_image(foreground_flow.permute(1, 2, 0).cpu().numpy())

                cv2.imwrite(path.join(self.save_dir, f"{epoch:03}/flow/{l:02}/{img_name}"), foreground_flow_img)
                cv2.imwrite(path.join(self.save_dir, f"{epoch:03}/foreground/{l:02}/{img_name}"), foreground_img)
                cv2.imwrite(path.join(self.save_dir, f"{epoch:03}/alpha/{l:02}/{img_name}"), alpha_img)

        return