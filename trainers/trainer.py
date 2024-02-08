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


class trainer_base():
    def __init__(
        self,
        data_dir: str = "data/",
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

    def epoch_loop(self):
        pass

    def train(self):
        pass

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
        self.train_format_csv_path = os.path.join(kfold_dir,
                                                  f"train_seed{kfold_seed}_kfold_{kfold_index}.csv")
        self.val_format_csv_path = os.path.join(kfold_dir,
                                                f"seed_seed{kfold_seed}_kfold_{kfold_index}.csv")
