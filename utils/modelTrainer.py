import os
import datetime
import argparse
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter


from utils.utils import create_dirs

class ModelTrainer(object):

    def __init__(self, 
                 args, 
                 model,
                 extra_models, 
                 dataset,
                 optimizer,
                 train_function,
                 test_function,
                 demo_function):
        """
        General model trainer class

        Args:
            args (namespace): namespace containing hyperparameters and essential parameters
                    must contain the following variables:
                        out_dir (str): directory in which all output such as logfiles, weights, results, etc. will be saved
                        epochs (int): number of epochs for training
                        demo_freq (int): every {demo_freq} epochs the ModelTrainer will perform a demo function. Will not perform demos if demo_freq is None
            models (torch.nn.Module): model that is trained
            extra_models (list[torch.nn.Module]): list of supplementary model instances that are necessary for training
            dataset (torch.utils.data.DataSet): Dataset instance to create dataloaders
            optimizer (torch.optim.Optimizer): optimizer instance for updating model parameters
            train_function (function): function object that handles training steps of model
            test_function (function): function object that handles validation steps of model
            demo_function (function): function object that creates a demo of the models performance at the current timestep
        """

        self.train_function = train_function
        self.test_function = test_function
        self.demo_function = demo_function

        self.model = model
        self.extra_models = extra_models
        self.optimizer = optimizer

        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.create_dataloaders(dataset)

        # initialize logging
        self.data_dir = args.data_dir
        self.demo_frequency = args.demo_freq
        self.experiment_dir = os.path.join(
            args.log_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.checkpoint_dir = os.path.join(
            self.experiment_dir, 'checkpoints')
        self.demo_dir = os.path.join(
            self.experiment_dir, "demo")
        create_dirs([self.experiment_dir, self.checkpoint_dir, self.demo_dir])
        self.summary_writer = SummaryWriter(self.experiment_dir)


    def create_dataloaders(self, dataset, train_val_split = 0.8):
        """
        Create the train and validation dataloaders from the given dataset

        Args:
            dataset (torch.utils.data.DataSet): dataset instance to be used for creating the dataloaders
            train_val_split (float): float denoting what fraction of the dataset is used for training. The rest is used for validation.
                                        (this should be defined on the interval [0, 1])
        
        Returns:
            train_loader (torch.utils.data.DataLoader): dataloader instance for training
            validation_loader (torch.utils.data.DataLoader): dataloader instance for validation
        """

        train_valid_split = int(train_val_split * len(dataset))

        train_set = torch.utils.data.dataset.Subset(dataset, np.arange(train_valid_split))
        valid_set = torch.utils.data.dataset.Subset(dataset, np.arange(train_valid_split, len(dataset)))

        self.train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
        self.valid_loader = torch.utils.data.DataLoader(
            valid_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            drop_last=False
        )

        return

    def train(self):
        """
        Train the model
        """
        

        # Tracking variables for finding best model
        best_epoch_performance = float('inf')
        best_epoch_idx = 0
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch : {epoch}")
            
            # Training epoch
            train_out = self.train_epoch()

            # Validation epoch
            valid_out = self.test_epoch()

            # Logging to TensorBoard
            for quantity in train_out.keys():
                
                if quantity == "performance":
                    continue

                self.summary_writer.add_scalars(
                    quantity, {"train": train_out[quantity], "val": valid_out[quantity]}, epoch)

            # Saving best model
            if valid_out["performance"] < best_epoch_performance:
                best_epoch_performance = valid_out["performance"]
                best_epoch_idx = epoch
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "epoch.pt"))

            # create a test demo
            if self.demo_frequency is not None and epoch % self.demo_frequency == 0:
                self.demo_epoch(epoch)


    def train_epoch(self):
        """
        Train the model for one epoch and handle saving the output as well as logging
        """
        if len(self.extra_models) > 0:
            return self.train_function(self.model, self.extra_models, self.train_loader, self.optimizer)
        else:
            return self.train_function(self.model, self.train_loader, self.optimizer)


    def test_epoch(self):
        """
        Test the models performance on the validation set
        """
        if len(self.extra_models) > 0:
            return self.test_function(self.model, self.extra_models, self.valid_loader)
        else:
            return self.test_function(self.model, self.valid_loader)

    def demo_epoch(self, epoch):
        """
        Create a demo of the models performance
        """
        if len(self.extra_models) > 0:
            return self.demo_function(self.data_dir, self.demo_dir, self.model, self.extra_models, epoch)
        else:
            return self.demo_function(self.data_dir, self.demo_dir, self.model, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # relevant directories
    parser.add_argument("--result_dir", type=str, 
                        help="Root directory of results")

    parser.add_argument("--seed", default=42, type=int,
                        help="Seed to use for reproducing results")
    parser.add_argument("--epochs", default=10, type=int,
                        help="number of epochs for training")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batch size for the model")
    parser.add_argument("--num_workers", default=1, type=int,
                        help="Number of workers to use in the dataloaders")
    parser.add_argument("--demo_freq", default=5, type=int,
                        help="Frequency at which the demo fucntion is called")
