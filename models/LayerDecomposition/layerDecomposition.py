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
                 save_dir: str):
        super().__init__()

        self.dataloader = dataloader
        self.loss_module = loss_module
        self.net = network
        self.optimizer = Adam(self.net.parameters(), learning_rate)

        self.save_dir = save_dir
        self.save_freq = 10

    def train(self, n_epochs):

        for epoch in range(n_epochs):
            if epoch % self.save_freq == 0:
                create_dirs(path.join(self.save_dir, f"{epoch:03}/background"), 
                            path.join(self.save_dir, f"{epoch:03}/foreground"),
                            path.join(self.save_dir, f"{epoch:03}/alpha"),
                            path.join(self.save_dir, f"{epoch:03}/reconstruction"),
                            path.join(self.save_dir, f"{epoch:03}/flow"))
            print(f"Epoch: {epoch} / {n_epochs}")

            for i, (input, targets) in enumerate(self.dataloader):

                output = self.net(input)
                loss = self.loss_module(output, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if epoch % self.save_freq == 0:
                    self.visualize_and_save_output(output, i, epoch)

    def visualize_and_save_output(self, model_output, i, epoch):
        """
        Save the output of the model 
        """
        
        rgba_layers    = model_output["layers_rgba"]
        flow_layers    = model_output["layers_flow"]
        reconstruction = model_output["rgba_reconstruction"]

        background_rgb     = torch.clone(rgba_layers[0, 0, 0, :3]).detach()
        foreground_rgb     = torch.clone(rgba_layers[0, 0, 1, :3]).detach()
        reconstruction_rgb = torch.clone(reconstruction[0, 0, :3]).detach()

        foreground_flow = torch.clone(flow_layers[0, 0, 1]).detach()

        foreground_alpha = torch.clone(rgba_layers[0, 0, 1, 3]).detach()

        background_img     = (background_rgb.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255
        foreground_img     = (foreground_rgb.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255
        reconstruction_img = (reconstruction_rgb.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255
        alpha_img          = (foreground_alpha.detach().cpu().numpy() + 1) / 2. * 255

        foreground_flow_img = flow_to_image(foreground_flow.permute(1, 2, 0).cpu().numpy())

        cv2.imwrite(path.join(self.save_dir, f"{epoch:03}/flow/{i:05}.png"), foreground_flow_img)
        cv2.imwrite(path.join(self.save_dir, f"{epoch:03}/background/{i:05}.png"), background_img)
        cv2.imwrite(path.join(self.save_dir, f"{epoch:03}/foreground/{i:05}.png"), foreground_img)
        cv2.imwrite(path.join(self.save_dir, f"{epoch:03}/reconstruction/{i:05}.png"), reconstruction_img)
        cv2.imwrite(path.join(self.save_dir, f"{epoch:03}/alpha/{i:05}.png"), alpha_img)

        return