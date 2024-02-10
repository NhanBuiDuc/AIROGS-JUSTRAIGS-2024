import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecificityLoss(nn.Module):
    def __init__(self, threshold=0.5, alpha=1.0):
        super(SpecificityLoss, self).__init__()
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, y_true, y_pred_prob):
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(y_pred_prob, y_true)

        # Calculate specificity (True Negative Rate)
        tn = torch.sum((1 - y_true) * (1 - torch.round(y_pred_prob)))
        fp = torch.sum((1 - y_true) * torch.round(y_pred_prob))
        specificity = tn / (tn + fp)

        # Calculate the proximity to the desired specificity threshold
        proximity_loss = torch.abs(specificity - (1.0 - self.threshold))

        # Combine binary cross-entropy loss and proximity loss
        total_loss = bce_loss + self.alpha * proximity_loss

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
