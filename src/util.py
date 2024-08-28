import torch
import torch.nn.functional as F

def compute_margin(logits, targets):
    """
    Computes the margin between the true class and the second-highest class for each example.
    Only computes the margin for examples where the true class is the highest.

    Args:
    - logits (torch.Tensor): Tensor of shape (batch_size, num_classes) containing logits.
    - targets (torch.Tensor): Tensor of shape (batch_size,) containing the true class indices.

    Returns:
    - margin_prob (torch.Tensor): Average margin, computed in probability space.
    - num_examples (int): Number of examples where the true class is the highest logit.
    """

    # Compute probabilities using softmax
    probabilities = F.softmax(logits, dim=-1)
    
    # Get the top 2 logits and their corresponding indices
    top2_logits, top2_indices = torch.topk(logits, 2, dim=-1)

    # Extract the top logits
    top1_logits = top2_logits[:, 0]  # Logit for the highest class
    top2_logits = top2_logits[:, 1]  # Logit for the second-highest class

    # Compute the probabilities for the top 2 classes
    top1_probs = probabilities.gather(1, top2_indices[:, 0:1]).squeeze()  # Probability of the highest class
    top2_probs = probabilities.gather(1, top2_indices[:, 1:2]).squeeze()  # Probability of the second-highest class
    
    # Compute margins in probability space
    margin_prob = top1_probs - top2_probs

    # Mask to only consider examples where the true class is the highest logit
    true_class_is_highest = (logits.gather(1, targets.unsqueeze(1)) == top1_logits.unsqueeze(1)).squeeze()

    # Apply the mask to filter out invalid margins
    margin_prob = margin_prob[true_class_is_highest]

    return margin_prob.mean().item(), len(margin_prob)


# Example usage
if __name__ == "__main__":
    # Example usage
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 0.2]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    margin_prob, num_examples = compute_margin(logits, targets)
    print("Logits:", logits)
    print("Targets:", targets)
    print("Margin in Probability Space:", margin_prob)
    print("Number of Examples:", num_examples)
    # Expected output:
    # Margin in Probability Space: 0.4009554982185364
    # Number of Examples: 2
    