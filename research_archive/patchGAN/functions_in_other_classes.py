
#### These functions where implemented in the LayerDecomposition class

def visualize_and_save_improvement_output(self, input, frame_indices, epoch):
    """
    Save the improvement model output
    """

    create_dir(path.join(self.results_root, "decomposition/improvement", f"{epoch:03}"))

    model_input = input[:, 0, :3]
    model_input = torch.cat((model_input, torch.randn_like(model_input)), dim=1)

    improved_background = self.improvement_net.improve(model_input)

    batch_size = improved_background.shape[0]

    for b in range(batch_size):

        img = (improved_background[b].permute(1, 2, 0).detach().cpu().numpy() * .5 + .5) * 255

        img_name = f"{frame_indices[b]:05}.png"
        epoch_name = f"{epoch:03}" if isinstance(epoch, int) else epoch

        cv2.imwrite(path.join(self.results_root, "decomposition/improvement", epoch_name, img_name), img)

def run_improvement_training(self):
    """
    Training script for the improvement GAN       
    """

    for epoch in range(self.n_epochs_gan):

        t0 = datetime.now()
        
        for iteration, (input, ground_truth, frame_indices) in enumerate(self.improvement_loader):
            
            self.improvement_net.update_cycle(input, ground_truth, iteration + epoch * len(self.improvement_loader))

            if epoch % self.save_freq == 0:
                self.visualize_and_save_improvement_output(input, frame_indices, epoch)

        t1 = datetime.now()
        print(f"Epoch: {epoch} / {self.n_epochs_gan - 1} done in {(t1 - t0).total_seconds()} seconds")