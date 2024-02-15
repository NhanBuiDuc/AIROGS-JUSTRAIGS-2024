from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, WeightedRandomSampler, BatchSampler
from torchvision.transforms import transforms
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.utils.class_weight import compute_class_weight
from datasets.AIROGS_dataset import Airogs_Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import resnet50, swin_v2_b, mobilenet_v3_large
from csv_logger import CsvLogger
import logging
from time import sleep
from loss.custom_loss import SpecificityLoss


class trainer_base():
    def __init__(
        self,
        data_dir: str = "data",
        num_epoch: int = 50,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.2),
        image_size: int = 256,
        num_class: int = 2,
        class_name: list = ["NRG", "RG"],
        kfold_seed: int = 42,
        kfold_index: int = 0,
        batch_size: int = 16,
        num_workers: int = 0,
        is_transform: bool = True,
        device: str = "cuda",
        early_stop_max_patient: int = 50
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
        self.device = device
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
            self.data_dir, "AIROGS_2024", "preprocessed_images")
        self.train_gt_path = os.path.join(
            self.data_dir, "AIROGS_2024", "JustRAIGS_Train_labels.csv")

        self.train_gt_path = self.train_gt_path.replace("'", '"')

        self.prepare_data(kfold_index, kfold_seed)
        # self.loss_fn = torch.nn.BCELoss()
        self.loss_fn = SpecificityLoss(
            specificity=0.95, alpha=1.5, positive_confidence=0.8, device="cuda")
        self.desired_specificity = 0.95
        self.early_stop_max_patient = early_stop_max_patient
        self.logger = None

    def train_loop_start(self):
        self.train_logits_list = []
        self.train_Y_list = []
        self.val_logits_list = []
        self.val_Y_list = []
        self.val_current_best_sensitivity = 0.0
        self.patient_count = 0
        self.model = mobilenet_v3_large(weights=None, progress=True,
                                        num_classes=1)
        # checkpoint_path = 'logs/logs_seed_42_fold_0_epoch_6.pth'

        # # Load the model state dictionary from the checkpoint
        # checkpoint = torch.load(
        #     checkpoint_path)
        # self.model = checkpoint
        self.model.to(self.device)
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=10, threshold=0.00001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

    def train_loop(self):

        # self.model = resnet50(weights=None, progress=True,
        #                       num_classes=1)

        m = nn.Sigmoid()

        for epoch in range(self.num_epoch):
            self.model.train()  # Set the self.model to training mode
            train_loss = 0.0
            val_loss = 0.0
            with tqdm(total=len(self.train_dataloader), unit="batch", mininterval=0) as bar:
                bar.set_description(f"Epoch {epoch}")

                for Xbatch, ybatch in self.train_dataloader:
                    Xbatch = Xbatch.to(self.device)
                    ybatch = ybatch.to(self.device)

                    # Forward pass
                    y_logits = self.model(Xbatch)
                    y_logits = m(y_logits)
                    y_logits = y_logits.squeeze(1)
                    loss = self.loss_fn(y_logits, ybatch)
                    self.train_logits_list.append(y_logits)
                    self.train_Y_list.append(ybatch)
                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Update metrics
                    train_loss += loss.item()

                    # Update progress bar
                    bar.set_postfix(loss=train_loss / (bar.n + 1))
                    bar.update()

            # Calculate average training loss and accuracy
            avg_train_loss = train_loss / len(self.train_dataloader)

            # Evaluate the self.model on the test set
            self.model.eval()  # Set the self.model to evaluation mode

            with torch.no_grad():
                for Xbatch, ybatch in self.val_dataloader:

                    Xbatch = Xbatch.to(self.device)
                    ybatch = ybatch.to(self.device)

                    y_logits = self.model(Xbatch)
                    y_logits = m(y_logits)
                    y_logits = y_logits.squeeze(1)
                    loss = self.loss_fn(y_logits, ybatch)
                    self.val_logits_list.append(y_logits)
                    self.val_Y_list.append(ybatch)

                    # Update metrics
                    val_loss += loss.item()
                    self.scheduler.step(val_loss)
            # Calculate average training loss and accuracy
            avg_val_loss = val_loss / len(self.val_dataloader)
            # Print and log the metrics
            print(
                f"Epoch {epoch + 1}/{self.num_epoch}, Train Loss: {avg_train_loss:.4f}, Train Loss: {avg_val_loss:.2%}")
            if self.train_epoch_end(epoch, avg_train_loss, avg_val_loss) == True:
                break
            else:
                continue

    def train_epoch_end(self, epoch, avg_train_loss, avg_val_loss):
        train_merged_logits = torch.cat(self.train_logits_list, dim=0)
        train_merged_gt = torch.cat(self.train_Y_list, dim=0)
        val_merged_logits = torch.cat(self.val_logits_list, dim=0)
        val_merged_gt = torch.cat(self.val_Y_list, dim=0)

        train_merged_logits = train_merged_logits.detach().cpu().numpy()
        train_merged_gt = train_merged_gt.detach().cpu().numpy()
        val_merged_logits = val_merged_logits.detach().cpu().numpy()
        val_merged_gt = val_merged_gt.detach().cpu().numpy()
        # Compute the ROC curve
        train_fpr, train_tpr, train_thresholds = roc_curve(
            train_merged_gt, train_merged_logits)
        val_fpr, val_tpr, val_thresholds = roc_curve(
            val_merged_gt, val_merged_logits)
        # Desired specificity

        # Find the index of the threshold that is closest to the desired specificity
        train_threshold_idx = np.argmax(
            train_fpr >= (1 - self.desired_specificity))
        val_threshold_idx = np.argmax(
            val_fpr >= (1 - self.desired_specificity))
        # Get the corresponding threshold
        train_threshold = train_thresholds[train_threshold_idx]
        val_threshold = val_thresholds[val_threshold_idx]
        # Get the corresponding TPR (sensitivity)
        train_sensitivity = train_tpr[train_threshold_idx]
        val_sensitivity = val_tpr[val_threshold_idx]
        # Calculate the AUC (Area Under the Curve)
        train_roc_auc = auc(train_fpr, train_tpr)
        val_roc_auc = auc(val_fpr, val_tpr)
        train_target_count_zeros = np.count_nonzero(train_merged_gt == 0)
        train_target_count_ones = np.count_nonzero(train_merged_gt == 1)
        val_target_count_zeros = np.count_nonzero(val_merged_gt == 0)
        val_target_count_ones = np.count_nonzero(val_merged_gt == 1)

        # Get the predicted labels based on the threshold
        train_predicted_labels = (
            train_merged_logits >= train_threshold).astype(int)
        val_predicted_labels = (
            val_merged_logits >= val_threshold).astype(int)
        train_predicted_labels_50 = (
            train_merged_logits >= 50).astype(int)
        val_predicted_labels_50 = (
            val_merged_logits >= 50).astype(int)
        # Compute confusion matrix
        train_conf_matrix = confusion_matrix(
            train_merged_gt, train_predicted_labels)
        val_conf_matrix = confusion_matrix(
            val_merged_gt, val_predicted_labels)
        train_conf_matrix_50 = confusion_matrix(
            train_merged_gt, train_predicted_labels)
        val_conf_matrix_50 = confusion_matrix(
            val_merged_gt, val_predicted_labels)
        if val_sensitivity > self.val_current_best_sensitivity:
            self.best_sensitivity = self.val_current_best_sensitivity
            self.auc_at_best_sensitivity = val_roc_auc
            self.thresh_hold_at_best_sensitivity = val_threshold
            self.val_current_best_sensitivity = val_sensitivity

        # Calculate accuracy
        train_accuracy = accuracy_score(
            train_merged_gt, train_predicted_labels)

        # Calculate precision
        train_precision = precision_score(
            train_merged_gt, train_predicted_labels)

        # Calculate recall
        train_recall = recall_score(train_merged_gt, train_predicted_labels)

        # Calculate F1 score
        train_f1 = f1_score(train_merged_gt, train_predicted_labels)

        val_accuracy = accuracy_score(val_merged_gt, val_predicted_labels)

        # Calculate precision
        val_precision = precision_score(val_merged_gt, val_predicted_labels)

        # Calculate recall
        val_recall = recall_score(val_merged_gt, val_predicted_labels)

        # Calculate F1 score
        val_f1 = f1_score(val_merged_gt, val_predicted_labels)

        train_accuracy_50 = accuracy_score(
            train_merged_gt, train_predicted_labels_50)

        # Calculate precision
        train_precision_50 = precision_score(
            train_merged_gt, train_predicted_labels_50)
        # Calculate recall
        train_recall_50 = recall_score(
            train_merged_gt, train_predicted_labels_50)

        # Calculate F1 score
        train_f1_50 = f1_score(train_merged_gt, train_predicted_labels_50)

        val_accuracy_50 = accuracy_score(
            val_merged_gt, val_predicted_labels_50)

        # Calculate precision
        val_precision_50 = precision_score(
            val_merged_gt, val_predicted_labels_50)

        # Calculate recall
        val_recall_50 = recall_score(val_merged_gt, val_predicted_labels_50)

        # Calculate F1 score
        val_f1_50 = f1_score(val_merged_gt, val_predicted_labels_50)

        train_sensitivity_50 = len(
            train_conf_matrix_50[1][1] / (train_conf_matrix_50[1][1] + train_conf_matrix_50[1][0]))
        val_sensitivity_50 = len(
            val_conf_matrix_50[1][1] / (val_conf_matrix_50[1][1] + val_conf_matrix_50[1][0]))
        print("train/threshold: ", 50)
        print("train/confusion_matrix_50: ", train_conf_matrix_50)
        print("train/sensitivity: ", train_sensitivity_50)
        print("train/true_negative_50: ", train_conf_matrix_50[0][0])
        print("train/false_positive_50", train_conf_matrix_50[0][1])
        print("train/false_negative_50: ", train_conf_matrix_50[1][0])
        print("train/true_positive_50", train_conf_matrix_50[1][1])
        print("train/acc_50", train_accuracy_50)
        print("train/f1_50", train_f1)
        print("train/recall_50", train_recall_50)
        print("train/precision_50", train_precision_50)

        print("val/sensitivity_50: ", val_sensitivity_50)
        print("val/confusion_matrix_50: ", val_conf_matrix_50)
        print("val/true_negative_50: ", val_conf_matrix[0][0])
        print("val/false_positive_50", val_conf_matrix[0][1])
        print("val/false_negative_50: ", val_conf_matrix[1][0])
        print("val/true_positive_50", val_conf_matrix[1][1])
        print("val/acc_50", val_accuracy_50)
        print("val/f1_50", val_f1_50)
        print("val/recall_50", val_recall_50)
        print("val/precision_50", val_precision_50)
        print("val/loss_50", avg_val_loss_50)

        ####################################################
        print("train/threshold: ", train_threshold)
        print("train/target_count_zeros: ", train_target_count_zeros)
        print("train/target_count_ones: ", train_target_count_ones)
        print("train/confusion_matrix: ", train_conf_matrix)
        print("train/sensitivity: ", train_sensitivity)
        print("train/roc_auc: ", train_roc_auc)
        print("train/true_negative: ", train_conf_matrix[0][0])
        print("train/false_positive", train_conf_matrix[0][1])
        print("train/false_negative: ", train_conf_matrix[1][0])
        print("train/true_positive", train_conf_matrix[1][1])
        print("train/acc", train_accuracy)
        print("train/f1", train_f1)
        print("train/recall", train_recall)
        print("train/precision", train_precision)
        print("train/loss", avg_train_loss)

        print("val/sensitivity: ", val_sensitivity)
        print("val/roc_auc: ", val_roc_auc)
        print("val/threshold: ", val_threshold)
        print("val/target_count_zeros: ", val_target_count_zeros)
        print("val/target_count_ones: ", val_target_count_ones)
        # print("val/pred_count_zeros", pred_count_zeros)
        # print("val/pred_count_ones", pred_count_ones)
        print("val/confusion_matrix: ", val_conf_matrix)
        print("val/true_negative: ", val_conf_matrix[0][0])
        print("val/false_positive", val_conf_matrix[0][1])
        print("val/false_negative: ", val_conf_matrix[1][0])
        print("val/true_positive", val_conf_matrix[1][1])
        print("val/sensitivity_best", self.val_current_best_sensitivity)
        print("val/roc_auc_best", self.auc_at_best_sensitivity)
        print("val/thresh_hold_best", self.thresh_hold_at_best_sensitivity)
        print("val/acc", val_accuracy)
        print("val/f1", val_f1)
        print("val/recall", val_recall)
        print("val/precision", val_precision)
        print("val/loss", avg_val_loss)
        self.train_logits_list = []
        self.train_Y_list = []
        self.val_logits_list = []
        self.val_Y_list = []
        if not os.path.exists("logs"):
            os.makedirs("logs")
        self.logs_path = os.path.join(
            "logs", f"logs_seed_{self.kfold_seed}_fold_{self.kfold_index}")
        model_name = f"logs_seed_{self.kfold_seed}_fold_{self.kfold_index}_epoch_{epoch}.pth"
        model_path = os.path.join(
            "logs", model_name)
        if self.logger is not None:
            self.logger = CsvLogger(
                self.logs_path, separator=",", append=False)
        is_early_stop = self.early_stop(self.model, model_path)
        if not is_early_stop:
            self.trigger_scheduler()
            self.save_checkpoint(self.model, model_path)
        return is_early_stop

    def early_stop(self, model, model_path):
        self.patient_count += 1
        if self.patient_count >= self.early_stop_max_patient:
            self.save_checkpoint(model, model_path)
            return True
        else:
            return False

    def init_logger(self):
        filename = 'logs/log.csv'
        delimiter = ','
        level = logging.INFO
        custom_additional_levels = ['logs_a', 'logs_b', 'logs_c']
        fmt = f'%(asctime)s{delimiter}%(levelname)s{delimiter}%(message)s'
        datefmt = '%Y/%m/%d %H:%M:%S'
        max_size = 1024  # 1 kilobyte
        max_files = 4  # 4 rotating files
        header = ['date', 'level', 'value_1', 'value_2']

        # Creat logger with csv rotating handler
        csvlogger = CsvLogger(filename=filename,
                              delimiter=delimiter,
                              level=level,
                              add_level_names=custom_additional_levels,
                              add_level_nums=None,
                              fmt=fmt,
                              datefmt=datefmt,
                              max_size=max_size,
                              max_files=max_files,
                              header=header)

        # Log some records
        for i in range(10):
            csvlogger.logs_a([i, i * 2])
            sleep(0.1)

        # You can log list or string
        csvlogger.logs_b([1000.1, 2000.2])
        csvlogger.critical('3000,4000')

        # Log some more records to trigger rollover
        for i in range(50):
            csvlogger.logs_c([i * 2, float(i**2)])
            sleep(0.1)

        # Read and print all of the logs from file after logging
        all_logs = csvlogger.get_logs(evaluate=False)
        for log in all_logs:
            print(log)

    def trigger_scheduler(self):
        pass

    def save_checkpoint(self, model, model_path):
        # Save the entire model
        torch.save(model, model_path)

    def train_loop_end(self):
        self.self.model.savecheckpoint()

    def train(self):
        self.train_loop_start()
        self.train_loop()
        # self.train_loop_end()

    def eval(self):
        pass

    def test(self):
        pass

    def prepare_data(self, kfold_index, kfold_seed) -> None:
        if (kfold_index == 0):
            kfold_dir = os.path.join(
                self.data_dir, "AIROGS_2024", "fold_split_images", f"fold_{kfold_index}")
        elif (kfold_index == 1):
            kfold_dir = os.path.join(
                self.data_dir, "AIROGS_2024", "fold_split_images", f"fold_{kfold_index}")
        elif (kfold_index == 2):
            kfold_dir = os.path.join(
                self.data_dir, "AIROGS_2024", "fold_split_images", f"fold_{kfold_index}")
        elif (kfold_index == 3):
            kfold_dir = os.path.join(
                self.data_dir, "AIROGS_2024", "fold_split_images", f"fold_{kfold_index}")
        elif (kfold_index == 4):
            kfold_dir = os.path.join(
                self.data_dir, "AIROGS_2024", "fold_split_images", f"fold_{kfold_index}")
        elif (kfold_index == 5):
            kfold_dir = os.path.join(
                self.data_dir, "AIROGS_2024", "fold_split_images", f"fold_{kfold_index}")

        kfold_dir = kfold_dir.replace("\\", "/")
        kfold_dir = kfold_dir.replace("//", "/")
        train_format_csv_path = os.path.join(kfold_dir,
                                             f"train_seed_{kfold_seed}_kfold_{kfold_index}.csv")

        train_format_csv_path = train_format_csv_path.replace("\\", "/")
        train_format_csv_path = train_format_csv_path.replace("//", "/")
        val_format_csv_path = os.path.join(kfold_dir,
                                           f"val_seed_{kfold_seed}_kfold_{kfold_index}.csv")
        val_format_csv_path = val_format_csv_path.replace("\\", "/")
        val_format_csv_path = val_format_csv_path.replace("//", "/")
        self.train_df = pd.read_csv(train_format_csv_path, delimiter=",")
        self.train_image_name_list = self.train_df["Eye ID"]
        self.train_label = self.train_df["Final Label"]
        self.val_df = pd.read_csv(val_format_csv_path, delimiter=",")
        self.val_image_path = self.val_df["Eye ID"]
        self.val_label = self.val_df["Final Label"]

        self.train_dataset = Airogs_Dataset(self.train_image_name_list.tolist(), self.train_label, self.class_name, len(
            self.train_label), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, True, self.image_size)

        self.val_dataset = Airogs_Dataset(self.train_image_name_list.tolist(), self.val_label, self.class_name, len(
            self.val_label), self.data_dir, self.train_image_path, self.is_transform, self.train_transforms, self.val_transforms, False, self.image_size)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=True)
