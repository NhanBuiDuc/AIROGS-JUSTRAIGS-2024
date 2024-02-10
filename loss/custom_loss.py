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
            torch.clamp(y_true * (threshold - y_pred_prob)) +
            torch.clamp((1 - y_true) * (y_pred_prob - threshold))
        )

        confidence_loss = torch.abs(positive_confidence - threshold)

        total_loss = self.alpha * logits_loss + bce_loss + confidence_loss
        return total_loss
# # Example usage
# desired_specificity = 0.95
# alpha = 1.0  # Adjust the weight of the proximity loss

# # Assuming y_true and y_pred_prob are your true labels and predicted probabilities
# y_true = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float32)
# y_pred_prob = torch.tensor([0.1, 0.8, 0.7, 0.2, 0.9], dtype=torch.float32)

# # Using the specified threshold
# threshold_at_desired_specificity = 0.8186508

# # Creating an instance of the custom loss
# loss_fn = SpecificityLoss(threshold_at_desired_specificity, alpha)

# # Applying the custom loss
# loss = loss_fn(y_true, y_pred_prob)

# print("Custom Loss:", loss.item())
