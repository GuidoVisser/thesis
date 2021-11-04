from typing import Union
import cv2
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, dataloader
from torch.optim import Adam
from os import path

from torch.utils.tensorboard.writer import SummaryWriter

from utils.utils import create_dirs
from models.third_party.RAFT.utils.flow_viz import flow_to_image


class LayerDecompositer(nn.Module):
    def __init__(self,
                 dataloader: DataLoader,
                 loss_module: nn.Module,
                 network: nn.Module,
                 memory_net: nn.Module,
                 summary_writer: SummaryWriter,
                 learning_rate: float,
                 mem_learning_rate: float,
                 mask_bootstrap_rolloff: int,
                 mask_loss_l1_rolloff: int,
                 results_root: str,
                 batch_size: int,
                 n_epochs: int,
                 save_freq: int):
        super().__init__()

        self.dataloader = dataloader
        self.loss_module = loss_module
        self.net = network
        self.memory_net = memory_net
        self.learning_rate = learning_rate
        self.mem_learning_rate = mem_learning_rate

        self.results_root = results_root
        self.save_dir = f"{results_root}/decomposition"
        self.n_epochs = n_epochs
        self.save_freq = save_freq
        self.batch_size = batch_size
        self.mask_bootstrap_rolloff = mask_bootstrap_rolloff
        self.mask_loss_l1_rolloff = mask_loss_l1_rolloff
        self.writer = summary_writer

    def run_training(self):
        
        self.optimizer = Adam(self.net.parameters(), self.learning_rate)
        self.memory_optimizer = Adam(self.memory_net.parameters(), self.mem_learning_rate)

        for epoch in range(self.n_epochs):
            
            if epoch % self.save_freq == 0:
                self.create_save_dirs(epoch)
            
            if epoch == self.mask_loss_l1_rolloff:
                self.loss_module.lambda_alpha_l1 = 0.

            if epoch == self.mask_bootstrap_rolloff:
                self.loss_module.lambda_mask_bootstrap = 0
                self.net.module.use_context = True

            jitter_params = self.dataloader.dataset.prepare_jitter_parameters()

            if self.net.module.use_context:
                self.memory_optimizer.zero_grad()
                self.memory_net.module.set_global_contexts(jitter_params)
                self.net.module.contexts = self.memory_net.module.global_context

            if torch.cuda.device_count() <= 1:
                print(f"Epoch: {epoch} / {self.n_epochs - 1}")
            
            for iteration, (input, targets) in enumerate(self.dataloader):

                self.optimizer.zero_grad()
                output = self.net(input)

                # set targets to the same device as the output
                device = next(iter(output.values())).get_device()
                targets = {k:v.to(device) for (k, v) in targets.items()}

                loss, loss_values = self.loss_module(output, targets)
                global_step = iteration + epoch*len(self.dataloader)
                self.writer.add_scalars("losses", loss_values, global_step=global_step)

                loss.backward(retain_graph=(iteration < len(self.dataloader) - 1))
                self.optimizer.step()

                if epoch % self.save_freq == 0:
                    frame_indices = input["index"][:, 0].tolist()
                    self.visualize_and_save_output(output, targets, frame_indices, epoch)

            self.memory_optimizer.step()

        torch.save(self.net.state_dict(), path.join(self.results_root, "reconstruction_weights.pth"))
        torch.save(self.memory_net.state_dict(), path.join(self.results_root, "memory_weights.pth"))

    @torch.no_grad()
    def decomposite(self):

        self.create_save_dirs("inference")

        jitter_params = self.dataloader.dataset.prepare_jitter_parameters()

        self.memory_net.module.set_global_contexts(jitter_params)
        self.net.module.contexts = self.memory_net.module.global_context

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
            output["layers_rgba"] = self.transfer_detail(reconstruction[:, :, :3], rgba_layers, gt_image)

            # Save results
            frame_indices = input["index"][:, 0].tolist()
            self.visualize_and_save_output(output, targets, frame_indices, "inference")

    def visualize_and_save_output(self, model_output, targets, frame_indices, epoch):
        """
        Save the output of the model 
        """

        rgba_layers    = model_output["layers_rgba"]
        flow_layers    = model_output["layers_flow"]
        reconstruction = model_output["rgba_reconstruction"]
        
        background_offset = model_output["background_offset"]
        brightness_scale  = model_output["brightness_scale"]

        gt_rgb = targets["rgb"]

        context_volumes = model_output["context_volumes"]

        # NOTE: current_batch_size is not necessarily the same as self.batch_size at the end of an epoch
        current_batch_size, _, n_layers, _, _, _ = rgba_layers.shape 

        for b in range(current_batch_size):

            if frame_indices[b] == 0:
                timesteps = [0, 1]
            else:
                timesteps = [1]
            
            for t in timesteps:

                # background
                background_rgb     = torch.clone(rgba_layers[b, t, 0, :3]).detach()
                reconstruction_rgb = torch.clone(reconstruction[b, t, :3]).detach()
                gt_rgb_batch       = gt_rgb[b, t]

                background_img     = cv2.cvtColor((background_rgb.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255, cv2.COLOR_RGB2BGR)
                reconstruction_img = cv2.cvtColor((reconstruction_rgb.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255, cv2.COLOR_RGB2BGR)
                gt_rgb_img         = cv2.cvtColor((gt_rgb_batch.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2. * 255, cv2.COLOR_RGB2BGR)

                background_offset_img = torch.clone(background_offset[b, t]).detach().cpu().permute(1, 2, 0).numpy()
                brightness_scale_img  = (torch.clone(brightness_scale[b, t]).detach().permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255

                img_name = f"{(frame_indices[b] + t):05}.png"
                epoch_name = f"{epoch:03}" if isinstance(epoch, int) else epoch
                cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/background/{img_name}"), background_img)
                cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/reconstruction/{img_name}"), reconstruction_img)
                cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/ground_truth/{img_name}"), gt_rgb_img)

                cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/background_offset/{img_name}"), flow_to_image(background_offset_img, convert_to_bgr=True))
                cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/brightness_scale/{img_name}"), brightness_scale_img)

                # save context volumes
                if isinstance(context_volumes, torch.Tensor):
                    torch.save(context_volumes[b, t].detach().cpu(), path.join(self.save_dir, f"{epoch_name}/context_volumes/raw/{path.splitext(img_name)[0]}.pth"))

                for l in range(1, n_layers):
                    create_dirs(path.join(self.save_dir, f"{epoch_name}/foreground/{l:02}"),
                                path.join(self.save_dir, f"{epoch_name}/alpha/{l:02}"),
                                path.join(self.save_dir, f"{epoch_name}/flow/{l:02}"))
                    foreground_rgba     = torch.clone(rgba_layers[b, t, l]).detach()
                    foreground_flow    = torch.clone(flow_layers[b, t, l]).detach()
                    foreground_alpha   = torch.clone(rgba_layers[b, t, l, 3]).detach()

                    foreground_img      = cv2.cvtColor((foreground_rgba.permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255, cv2.COLOR_RGBA2BGRA)
                    alpha_img           = (foreground_alpha.cpu().numpy() + 1) / 2. * 255
                    foreground_flow_img = flow_to_image(foreground_flow.permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)

                    cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/flow/{l:02}/{img_name}"), foreground_flow_img)
                    cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/foreground/{l:02}/{img_name}"), foreground_img)
                    cv2.imwrite(path.join(self.save_dir, f"{epoch_name}/alpha/{l:02}/{img_name}"), alpha_img)

    def create_save_dirs(self, epoch):

        epoch_name = f"{epoch:03}" if isinstance(epoch, int) else epoch
        create_dirs(path.join(self.save_dir, f"{epoch_name}/background"), 
                    path.join(self.save_dir, f"{epoch_name}/foreground"),
                    path.join(self.save_dir, f"{epoch_name}/alpha"),
                    path.join(self.save_dir, f"{epoch_name}/reconstruction"),
                    path.join(self.save_dir, f"{epoch_name}/ground_truth"),
                    path.join(self.save_dir, f"{epoch_name}/flow"),
                    path.join(self.save_dir, f"{epoch_name}/background_offset"),
                    path.join(self.save_dir, f"{epoch_name}/brightness_scale"),
                    path.join(self.save_dir, f"{epoch_name}/context_volumes/raw"))

    def transfer_detail(self, reconstruction, rgba_layers, gt_image):
        residual = gt_image - reconstruction

        transmission_composite = torch.zeros_like(gt_image[:, :, 0:1])
        rgba_with_detail = rgba_layers

        n_layers = rgba_layers.shape[2]

        for i in range(n_layers - 1, 0, -1):
            layer_transmission = 1 - transmission_composite
            rgba_with_detail[:, :, i, :3] += layer_transmission * residual
            layer_alpha = rgba_layers[:, :, i, 3:4] * .5 + .5
            transmission_composite = layer_alpha + (1 - layer_alpha) * transmission_composite
        
        return torch.clamp(rgba_with_detail, -1, 1)