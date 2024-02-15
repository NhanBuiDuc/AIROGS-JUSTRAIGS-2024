import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score, auc
import numpy as np


class SpecificityLoss(nn.Module):
    def __init__(self, specificity=0.95, alpha=1.5, positive_confidence=0.8, device="cuda"):
        super(SpecificityLoss, self).__init__()
        self.specificity = specificity
        self.alpha = alpha
        self.positive_confidence = positive_confidence
        self.device = device

    def forward(self, y_pred_prob, y_true):

        positive_confidence = torch.tensor(
            self.positive_confidence, dtype=torch.float32)
        positive_confidence.to(self.device)
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(y_pred_prob, y_true)

        logits = y_pred_prob.detach().cpu().numpy()
        gt = y_true.detach().cpu().numpy()
        fpr, tpr, thresholds = roc_curve(
            gt, logits)
        threshold_idx = np.argmax(
            fpr >= (1 - self.specificity))
        threshold = thresholds[threshold_idx]
        logits_loss = (1.0 / len(y_true)) * torch.sum(
            torch.clamp(y_true * (threshold - y_pred_prob), min=0) +
            torch.clamp((1 - y_true) * (y_pred_prob - threshold), min=0)
        )

        confidence_loss = torch.abs(positive_confidence - threshold)

        total_loss = self.alpha * logits_loss + confidence_loss
        return total_loss


class F1DoubleSoftLoss(nn.Module):
    def __init__(self, mean):
        super(F1DoubleSoftLoss, self).__init__()
        self.reduction = mean

    def forward(self, y_pred_prob, y_true):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.
        This version uses the computation of soft-F1 for both positive and negative class for each label.

        Args:
            y (torch.FloatTensor): targets array of shape (BATCH_SIZE, N_LABELS), including 0. and 1.
            y_hat (torch.FloatTensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar): value of the cost function for the batch
        """

        # dtype = y_hat.dtype
        # y = y.to(dtype)

        # FloatTensor = torch.cuda.FloatTensor
        # y = FloatTensor(y)
        # y_hat = FloatTensor(y_hat)

        tp = (y_pred_prob * y_true).sum(dim=0)  # soft
        fp = (y_pred_prob * (1-y_true)).sum(dim=0)  # soft
        fn = ((1-y_pred_prob) * y_true).sum(dim=0)  # soft
        tn = ((1-y_pred_prob) * (1-y_true)).sum(dim=0)  # soft

        soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
        soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
        # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
        cost_class1 = 1 - soft_f1_class1
        # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
        cost_class0 = 1 - soft_f1_class0
        # take into account both class 1 and class 0
        cost = 0.5 * (cost_class1 + cost_class0)

        if self.reduction == 'none':
            return cost

        if self.reduction == 'mean':
            macro_cost = cost.mean()
            return macro_cost


def macro_double_soft_f1(y, y_hat, reduction='mean'):  # Written in PyTorch
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (torch.FloatTensor): targets array of shape (BATCH_SIZE, N_LABELS), including 0. and 1.
        y_hat (torch.FloatTensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar): value of the cost function for the batch
    """

    # dtype = y_hat.dtype
    # y = y.to(dtype)

    # FloatTensor = torch.cuda.FloatTensor
    # y = FloatTensor(y)
    # y_hat = FloatTensor(y_hat)

    tp = (y_hat * y).sum(dim=0)  # soft
    fp = (y_hat * (1-y)).sum(dim=0)  # soft
    fn = ((1-y_hat) * y).sum(dim=0)  # soft
    tn = ((1-y_hat) * (1-y)).sum(dim=0)  # soft

    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
    # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class1 = 1 - soft_f1_class1
    # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost_class0 = 1 - soft_f1_class0
    # take into account both class 1 and class 0
    cost = 0.5 * (cost_class1 + cost_class0)

    if reduction == 'none':
        return cost

    if reduction == 'mean':
        macro_cost = cost.mean()
        return macro_cost
