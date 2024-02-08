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
        image_size: int = 512,
        num_class: int = 2,
        class_name: list = ["NRG", "RG"],
        kfold_seed: int = 111,
        kfold_index: int = 0,
        batch_size: int = 16,
        num_workers: int = 0,
        is_transform=True,
        balance_data=True,
        binary_unbalance_train_ratio=100
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
        self.balance_data = balance_data
        self.train_image_path = os.path.join(
            self.data_dir, "ISBI_2024/resize_512_images/")
        self.train_gt_path = os.path.join(
            self.data_dir, "ISBI_2024", "JustRAIGS_Train_labels.csv")
        self.geo_aug_images = os.path.join(
            self.data_dir, "ISBI_2024", "geo_aug_images")
        self.color_aug_images = os.path.join(
            self.data_dir, "ISBI_2024", "color_aug_images")

        self.train_gt_path = self.train_gt_path.replace("\\", "/")

        self.binary_unbalance_train_ratio = binary_unbalance_train_ratio
        self.prepare_data()
        self.setup()

    def epoch_loop(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def test(self):
        pass

    def prepare_data(self) -> None:
        # Load the CSV file into a pandas DataFrame
        self.train_gt_pdf = pd.read_csv(self.train_gt_path, delimiter=';')
        # self.train_gt_pdf = self.train_gt_pdf[:100]
        self.train_image_name = self.train_gt_pdf["Eye ID"]
        self.train_label_list = self.train_gt_pdf.iloc[:, 1:].apply(
            lambda row: {col.lower(): row[col] for col in self.train_gt_pdf.columns[1:]}, axis=1).tolist()

        self.class_distribution = self.calculate_class_distribution()
        # # Calculate the number of samples for each split
        self.total_samples = int(sum(self.class_distribution.values()))

    def setup(self) -> None:

        if not self.data_train and not self.data_val:
            if not self.balance_data:
                input_data = self.train_gt_pdf['Eye ID']
                labels = self.train_gt_pdf['Final Label']
                # Map class labels to numerical values
                class_to_numeric = {class_label: idx for idx,
                                    class_label in enumerate(self.class_name)}

                # Transform labels into numerical format (0 or 1)
                labels_numeric = [class_to_numeric[label] for label in labels]
                labels_numeric = np.array(labels_numeric)
                # Choose fold to train on
                kf = KFold(n_splits=5,
                           shuffle=True, random_state=self.kfold_seed)

                all_splits = [k for k in kf.split(input_data, labels_numeric)]

                train_indexes, val_indexes = all_splits[self.kfold_index]

                # Count the number of samples in class 1 in the training set
                train_input_data = input_data[train_indexes]
                train_label_data = labels_numeric[train_indexes]

                val_input_data = input_data[val_indexes]
                val_label_data = labels_numeric[val_indexes]

                train_class_counts = np.bincount(train_label_data)
                val_class_counts = np.bincount(val_label_data)

                print("train/class_zeros_count: ", train_class_counts[0])
                print("train/class_ones_count: ", train_class_counts[1])
                print("val/class_zeros_count: ",
                      val_class_counts[0])
                print("val/class_ones_count: ", val_class_counts[1])

                # # Calculate class weights
                # train_class_weights = 1. / \
                #     torch.tensor(train_class_counts, dtype=torch.float)
                # # Map class labels to indices
                # train_class_to_index = {
                #     self.class_name[i]: i for i in range(len(self.class_name))}

                # train_label_indices = [train_class_to_index[label]
                #                        for label in train_label_data]

                # # Assign weights to each sample in the validation set
                # train_weights = train_class_weights[train_label_indices]

                # Assuming you have WeightedRandomSampler, you can use it like this:
                self.weighted_sampler_train = WeightedRandomSampler(
                    weights=[(0.8*self.batch_size) // 100,
                             (0.2*self.batch_size) // 100],
                    num_samples=len(train_label_data),
                    replacement=False
                )

                # Calculate class weights
                # val_class_weights = 1. / \
                #     torch.tensor(val_class_counts, dtype=torch.float)

                # Assuming you have WeightedRandomSampler, you can use it like this:
                self.weighted_sampler_val = WeightedRandomSampler(
                    weights=[(0.8*self.batch_size) // 100,
                             (0.2*self.batch_size) // 100],
                    num_samples=len(val_label_data),
                    replacement=False
                )

                self.data_train = Airogs_Dataset(
                    combined_train_data, combined_label_data, self.class_name, len(combined_train_data), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, is_training=True, image_size=self.image_size)

                self.data_val = Airogs_Dataset(
                    val_input_data.tolist(), val_label_data.tolist(), self.class_name, len(val_input_data), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, is_training=False, image_size=self.image_size)
            else:
                if self.balance_data:
                    input_data = self.train_gt_pdf['Eye ID']
                    labels = self.train_gt_pdf['Final Label']
                    # Map class labels to numerical values
                    class_to_numeric = {class_label: idx for idx,
                                        class_label in enumerate(self.class_name)}

                    # Transform labels into numerical format (0 or 1)
                    labels_numeric = [class_to_numeric[label]
                                      for label in labels]
                    labels_numeric = np.array(labels_numeric)
                    # Choose fold to train on
                    kf = KFold(n_splits=5,
                               shuffle=True, random_state=self.kfold_seed)

                    all_splits = [k for k in kf.split(
                        input_data, labels_numeric)]

                    train_indexes, val_indexes = all_splits[self.kfold_index]

                    # Count the number of samples in class 1 in the training set
                    train_input_data = input_data[train_indexes]
                    train_label_data = labels_numeric[train_indexes]

                    val_input_data = input_data[val_indexes]
                    val_label_data = labels_numeric[val_indexes]

                    train_class_counts = np.bincount(train_label_data)
                    val_class_counts = np.bincount(val_label_data)

                    print("original_train/class_zeros_count: ",
                          train_class_counts[0])
                    print("original_train/class_ones_count: ",
                          train_class_counts[1])
                    print("original_val/class_zeros_count: ",
                          val_class_counts[0])
                    print("original_val/class_ones_count: ",
                          val_class_counts[1])

                    # # Calculate class weights
                    # train_class_weights = 1. / \
                    #     torch.tensor(train_class_counts, dtype=torch.float)
                    # # Map class labels to indices
                    # train_class_to_index = {
                    #     self.class_name[i]: i for i in range(len(self.class_name))}

                    # train_label_indices = [train_class_to_index[label]
                    #                        for label in train_label_data]

                    # # Assign weights to each sample in the validation set
                    # train_weights = train_class_weights[train_label_indices]

                    # Assuming you have WeightedRandomSampler, you can use it like this:
                    self.weighted_sampler_train = WeightedRandomSampler(
                        weights=[(0.8*self.batch_size) // 100,
                                 (0.2*self.batch_size) // 100],
                        num_samples=len(train_label_data),
                        replacement=False
                    )

                    # Calculate class weights
                    # val_class_weights = 1. / \
                    #     torch.tensor(val_class_counts, dtype=torch.float)

                    # Assuming you have WeightedRandomSampler, you can use it like this:
                    self.weighted_sampler_val = WeightedRandomSampler(
                        weights=[(0.8*self.batch_size) // 100,
                                 (0.2*self.batch_size) // 100],
                        num_samples=len(val_label_data),
                        replacement=False
                    )
                    original_train_input_data = train_input_data.tolist()
                    original_train_label_data = train_label_data.tolist()

                    geo_images = [f for f in os.listdir(
                        self.geo_aug_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    color_images = [f for f in os.listdir(
                        self.color_aug_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    combined_train_data = original_train_input_data + geo_images + color_images

                    # Assuming label_data is all 1 for the length of the new images
                    combined_label_data = original_train_label_data + \
                        [1] * (len(geo_images) + len(color_images))

                    print("augmented_train/class_ones_count: ",
                          train_class_counts[1] + len(geo_images) + len(color_images))

                    self.data_train = Airogs_Dataset(
                        combined_train_data, combined_label_data, self.class_name, len(combined_train_data), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, is_training=True, image_size=self.image_size)

                    self.data_val = Airogs_Dataset(
                        val_input_data.tolist(), val_label_data.tolist(), self.class_name, len(val_input_data), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, is_training=False, image_size=self.image_size)
