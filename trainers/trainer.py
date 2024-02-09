from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, WeightedRandomSampler, BatchSampler
from torchvision.transforms import transforms
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.class_weight import compute_class_weight
from datasets.AIROGS_dataset import Airogs_Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import resnet50


class trainer_base():
    def __init__(
        self,
        data_dir: str = "data/",
        num_epoch: int = 50,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.2),
        image_size: int = 256,
        num_class: int = 2,
        class_name: list = ["NRG", "RG"],
        kfold_seed: int = 42,
        kfold_index: int = 0,
        batch_size: int = 16,
        num_workers: int = 0,
        is_transform=True,
    ) -> None:
        torch.backends.bottleneck = True
        train_trans = transforms.Compose(
            [transforms.ToTensor(), transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.5, hue=0.5),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        val_trans = transforms.Compose(
            [transforms.ToTensor(), transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.5, hue=0.5),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        trans1 = transforms.Compose([
            # transforms.RandomCrop(256),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        trans2 = transforms.Compose([
            # transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor()])

        trans3 = transforms.Compose([
            # transforms.RandomCrop(256),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor()])

        self.train_transforms = train_trans
        self.val_transforms = val_trans
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_ = batch_size
        self.data_dir = data_dir
        self.num_epoch = num_epoch
        self.image_size = image_size
        self.class_name = class_name
        self.num_class = num_class
        self.batch_size = batch_size
        self.kfold_seed = kfold_seed
        self.kfold_index = kfold_index
        self.training_split = train_val_test_split[0]
        self.validation_split = train_val_test_split[1]
        self.is_transform = is_transform
        self.num_workers = num_workers
        self.train_image_path = os.path.join(
            self.data_dir, "ISBI_2024/preprocessed_images/")
        self.train_gt_path = os.path.join(
            self.data_dir, "ISBI_2024", "JustRAIGS_Train_labels.csv")

        self.train_gt_path = self.train_gt_path.replace("\\", "/")

        self.prepare_data(kfold_index, kfold_seed)
        self.loss_fn = torch.nn.BCELoss()

    def train_loop_start(self):
        self.train_X_list = []
        self.train_Y_list = []
        self.val_X_list = []
        self.val_Y_list = []

    def train_loop(self):

        # Define your model, loss function, optimizer, and other parameters
        class YourModel(nn.Module):
            def __init__(self):
                super(YourModel, self).__init__()
                # Define your model layers here

            def forward(self, x):
                # Define the forward pass
                return x

        model = resnet50(weights=None, progress=True,
                         num_classes=1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(self.num_epoch):
            model.train()  # Set the model to training mode
            train_loss = 0.0
            train_acc = 0.0

            with tqdm(total=len(self.train_dataloader), unit="batch", mininterval=0) as bar:
                bar.set_description(f"Epoch {epoch}")

                for Xbatch, ybatch in self.train_dataloader:
                    # Forward pass
                    y_pred = model(Xbatch)
                    loss = self.loss_fn(y_pred, ybatch)
                    acc = (torch.argmax(y_pred, dim=1)
                           == ybatch).float().mean()

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update metrics
                    train_loss += loss.item()
                    train_acc += acc.item()

                    # Update progress bar
                    bar.set_postfix(loss=train_loss / (bar.n + 1),
                                    acc=train_acc / (bar.n + 1))
                    bar.update()

            # Calculate average training loss and accuracy
            avg_train_loss = train_loss / len(self.train_dataloader)
            avg_train_acc = train_acc / len(self.val_dataloader)

            # Evaluate the model on the test set
            model.eval()  # Set the model to evaluation mode
            test_acc = 0.0

            with torch.no_grad():
                for Xbatch, ybatch in self.val_dataloader:
                    y_pred = model(Xbatch)
                    acc = (torch.argmax(y_pred, dim=1)
                           == ybatch).float().mean()
                    test_acc += acc.item()

            avg_test_acc = test_acc / len(self.val_dataloader)

            # Print and log the metrics
            print(
                f"Epoch {epoch + 1}/{self.num_epoch}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2%}, Test Acc: {avg_test_acc:.2%}")

    def train(self):
        self.train_loop_start()
        self.train_loop()
        self.train_loop_end()

    def eval(self):
        pass

    def test(self):
        pass

    def prepare_data(self, kfold_index, kfold_seed) -> None:
        if (kfold_index == 0):
            kfold_dir = os.path.join(
                self.data_dir, "ISBI_2024/5kfold_split_images/", f"fold_{kfold_index}")
        elif (kfold_index == 1):
            kfold_dir = os.path.join(
                self.data_dir, "ISBI_2024/5kfold_split_images/", f"fold_{kfold_index}")
        elif (kfold_index == 2):
            kfold_dir = os.path.join(
                self.data_dir, "ISBI_2024/5kfold_split_images/", f"fold_{kfold_index}")
        elif (kfold_index == 3):
            kfold_dir = os.path.join(
                self.data_dir, "ISBI_2024/5kfold_split_images/", f"fold_{kfold_index}")
        elif (kfold_index == 4):
            kfold_dir = os.path.join(
                self.data_dir, "ISBI_2024/5kfold_split_images/", f"fold_{kfold_index}")
        elif (kfold_index == 5):
            kfold_dir = os.path.join(
                self.data_dir, "ISBI_2024/5kfold_split_images/", f"fold_{kfold_index}")
        train_format_csv_path = os.path.join(kfold_dir,
                                             f"train_seed_{kfold_seed}_kfold_{kfold_index}.csv")
        val_format_csv_path = os.path.join(kfold_dir,
                                           f"seed_seed_{kfold_seed}_kfold_{kfold_index}.csv")
        self.train_df = pd.read_csv(train_format_csv_path, delimiter=",")
        self.train_image_path = self.train_df["Eye ID"]
        self.train_label = self.train_df["Final Label"]
        self.val_df = pd.read_csv(val_format_csv_path, delimiter=",")
        self.val_image_path = self.val_df["Eye ID"]
        self.val_label = self.val_df["Final Label"]

        self.train_dataset = Airogs_Dataset(self.train_image_path, self.train_label, self.class_name, len(
            self.train_label), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, True, self.image_size)

        self.val_dataset = Airogs_Dataset(self.val_image_path, self.val_label, self.class_name, len(
            self.val_label), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, False, self.image_size)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=True)
