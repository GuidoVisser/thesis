import cv2
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from os import path
from datetime import datetime

from torch.utils.tensorboard.writer import SummaryWriter

from utils.utils import create_dir, create_dirs
from models.third_party.RAFT.utils.flow_viz import flow_to_image

class LayerDecompositer(nn.Module):
    def __init__(self,
                 dataloader: DataLoader,
                 loss_module: nn.Module,
                 network: nn.Module,
                 summary_writer: SummaryWriter,
                 learning_rate: float,
                 results_root: str,
                 batch_size: int,
                 n_epochs: int,
                 save_freq: int,
                 separate_bg: bool,
                 use_depth: bool):
        super().__init__()

        self.dataloader = dataloader
        self.loss_module = loss_module
        self.net = network
        self.learning_rate = learning_rate
        self.optimizer = Adam(self.net.parameters(), self.learning_rate)

        self.results_root = results_root
        self.save_dir = f"{results_root}/decomposition"
        self.n_epochs = n_epochs
        self.save_freq = save_freq
        self.batch_size = batch_size
        self.writer = summary_writer
        self.separate_bg = separate_bg
        self.use_depth = use_depth

    def run_training(self, start_epoch=0):
        
        for epoch in range(start_epoch, self.n_epochs):

            t0 = datetime.now()
            
            if epoch % self.save_freq == 0:
                self.create_save_dirs(f"intermediate/{epoch}")
            
            for iteration, (input, targets) in enumerate(self.dataloader):

                self.optimizer.zero_grad()
                output = self.net(input)

                # set targets to the same device as the output
                device = next(iter(output.values())).get_device()
                if device == -1:
                    device = "cpu"
                targets = {k:v.to(device) for (k, v) in targets.items()}

                loss, loss_values = self.loss_module(output, targets)
                global_step = iteration + epoch*len(self.dataloader)
                self.writer.add_scalars("losses", loss_values, global_step=global_step)

                loss.backward()
                self.optimizer.step()

                if epoch % self.save_freq == 0:
                    frame_indices = input["index"][:, 0].tolist()
                    self.visualize_and_save_output(output, targets, frame_indices, f"intermediate/{epoch}")

            self.loss_module.update_lambdas()

            if torch.cuda.device_count() <= 1:
                t1 = datetime.now()
                print(f"Epoch: {epoch} / {self.n_epochs - 1} done in {(t1 - t0).total_seconds()} seconds")

        torch.save(self.net.state_dict(), path.join(self.results_root, "reconstruction_weights.pth"))           

    @torch.no_grad()
    def decomposite(self):

        self.create_save_dirs("final")

        for (input, targets) in self.dataloader:

            # Forward pass through network
            output = self.net(input)

            # set targets to the same device as the output
            device = next(iter(output.values())).get_device()
            targets = {k:v.to(device) for (k, v) in targets.items()}

            # Do detail transfer
            reconstruction = output["rgba_reconstruction"]
            rgba_layers    = output["layers_rgba"]
            gt_image       = targets["rgb"]
            output["layers_rgba"] = self.transfer_detail(reconstruction[:, :3], rgba_layers, gt_image)

            # Save results
            frame_indices = input["index"][:, 0].tolist()
            self.visualize_and_save_output(output, targets, frame_indices, "final")

    def visualize_and_save_output(self, model_output, targets, frame_indices, epoch_name):
        """
        Save the output of the model 
        """

        rgba_layers    = model_output["layers_rgba"]
        flow_layers    = model_output["layers_flow"]
        if self.use_depth:
            depth_layers   = model_output["layers_depth"]
        reconstruction = model_output["rgba_reconstruction"]
        
        background_offset = model_output["background_offset"]
        brightness_scale  = model_output["brightness_scale"]

        if "full_static_bg" in model_output:
            full_static_bg = model_output["full_static_bg"]
            bg_plate = full_static_bg[0, :]
            background_plate_img  = cv2.cvtColor((bg_plate.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path.join(self.save_dir, epoch_name, "bg_plate.png"), background_plate_img)

        gt_rgb = targets["rgb"]

        # NOTE: current_batch_size is not necessarily the same as self.batch_size at the end of an epoch
        current_batch_size, n_layers, _, timesteps, _, _ = rgba_layers.shape 

        for b in range(current_batch_size):
            
            for t in range(timesteps):

                # background
                if self.separate_bg:
                    background_rgb_static    = torch.clone(rgba_layers[b, 0, :3, t]).detach()
                    background_rgb_dynamic   = torch.clone(rgba_layers[b, 1, :3, t]).detach()
                    background_alpha_dynamic = torch.clone(rgba_layers[b, 1, 3:, t]).detach()

                    # Go from tripmap to binary mask
                    background_alpha_dynamic = background_alpha_dynamic * .5 + .5

                    # Get the full background
                    background_rgb = (1 - background_alpha_dynamic) * background_rgb_static + background_alpha_dynamic * background_rgb_dynamic
                else:
                    background_rgb = torch.clone(rgba_layers[b, 0, :3, t]).detach()

                reconstruction_rgb = torch.clone(reconstruction[b, :3 , t]).detach()
                gt_rgb_batch       = gt_rgb[b, :, t]

                background_img        = cv2.cvtColor((background_rgb.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255, cv2.COLOR_RGB2BGR)
                if self.separate_bg:
                    background_img_static = cv2.cvtColor((background_rgb_static.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255, cv2.COLOR_RGB2BGR)
                reconstruction_img    = cv2.cvtColor((reconstruction_rgb.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255, cv2.COLOR_RGB2BGR)
                gt_rgb_img            = cv2.cvtColor((gt_rgb_batch.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2. * 255, cv2.COLOR_RGB2BGR)

                background_offset_img = torch.clone(background_offset[b, :, t]).detach().cpu().permute(1, 2, 0).numpy()
                brightness_scale_img  = (torch.clone(brightness_scale[b, :, t]).detach().permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255

                img_name = f"{(frame_indices[b] + t):05}.png"
                cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/background/{img_name}"), background_img)
                if self.separate_bg:
                    cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/background_static/{img_name}"), background_img_static)
                cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/reconstruction/{img_name}"), reconstruction_img)
                cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/ground_truth/{img_name}"), gt_rgb_img)

                cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/background_offset/{img_name}"), flow_to_image(background_offset_img[:, :, :2], convert_to_bgr=True))
                cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/brightness_scale/{img_name}"), brightness_scale_img)

                for l in range(1, n_layers):
                    create_dirs(path.join(self.save_dir, f"{epoch_name}/foreground/{l:02}"),
                                path.join(self.save_dir, f"{epoch_name}/alpha/{l:02}"),
                                path.join(self.save_dir, f"{epoch_name}/flow/{l:02}"))
                    foreground_rgba    = torch.clone(rgba_layers[b, l, :, t]).detach()
                    foreground_flow    = torch.clone(flow_layers[b, l, :, t]).detach()
                    foreground_alpha   = torch.clone(rgba_layers[b, l, 3, t]).detach()

                    foreground_img      = cv2.cvtColor((foreground_rgba.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255, cv2.COLOR_RGBA2BGRA)
                    alpha_img           = (foreground_alpha.cpu().numpy() + 1) / 2. * 255
                    foreground_flow_img = flow_to_image(foreground_flow.permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)

                    cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/flow/{l:02}/{img_name}"), foreground_flow_img)
                    cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/foreground/{l:02}/{img_name}"), foreground_img)
                    cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/alpha/{l:02}/{img_name}"), alpha_img)

                    if self.use_depth:
                        create_dir(path.join(self.save_dir, f"{epoch_name}/depth/{l:02}"))
                        foreground_depth   = torch.clone(depth_layers[b, l, 0, t]).detach()
                        depth_img           = (1 - ((foreground_depth.cpu().numpy() + 1) / 2.)) * 255
                        cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/depth/{l:02}/{img_name}"), depth_img)


    def create_save_dirs(self, epoch):

        epoch_name = f"{epoch:03}" if isinstance(epoch, int) else epoch
        create_dirs(path.join(self.save_dir, f"{epoch_name}/background"),
                    path.join(self.save_dir, f"{epoch_name}/foreground"),
                    path.join(self.save_dir, f"{epoch_name}/alpha"),
                    path.join(self.save_dir, f"{epoch_name}/reconstruction"),
                    path.join(self.save_dir, f"{epoch_name}/ground_truth"),
                    path.join(self.save_dir, f"{epoch_name}/flow"),
                    path.join(self.save_dir, f"{epoch_name}/depth"),
                    path.join(self.save_dir, f"{epoch_name}/background_offset"),
                    path.join(self.save_dir, f"{epoch_name}/brightness_scale"))
        if self.separate_bg:
            create_dirs(path.join(self.save_dir, f"{epoch_name}/background_static"))
        if self.use_depth:
            create_dirs(path.join(self.save_dir, f"{epoch_name}/depth"))

    def transfer_detail(self, reconstruction, rgba_layers, gt_image):
        residual = gt_image - reconstruction

        transmission_composite = torch.zeros_like(gt_image[:, :, 0:1])
        rgba_with_detail = rgba_layers

        n_layers = rgba_layers.shape[1]
        n_bg_layers = 2 if self.separate_bg else 1

        for i in range(n_layers - 1, n_bg_layers - 1, -1):
            layer_transmission = 1 - transmission_composite
            rgba_with_detail[:, i, :3] += layer_transmission * residual
            layer_alpha = rgba_layers[:, i, 3:4] * .5 + .5
            transmission_composite = layer_alpha + (1 - layer_alpha) * transmission_composite
        
        return torch.clamp(rgba_with_detail, -1, 1)