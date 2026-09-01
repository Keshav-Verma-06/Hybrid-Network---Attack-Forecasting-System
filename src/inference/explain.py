import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import numpy as np

class GRURiskWrapper(nn.Module):
    """Wraps the GRU model to return ONLY the risk logit for Captum integration."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        _, fut_logits, _, _ = self.model(x)
        return fut_logits[:, 0].unsqueeze(1)  # K=1 future risk

def explain_prediction(model, input_seq, feature_names, device):
    """Uses Integrated Gradients to find the most influential features."""
    wrapper = GRURiskWrapper(model).to(device)
    wrapper.eval()
    
    ig = IntegratedGradients(wrapper)
    
    input_seq.requires_grad_()
    
    # Calculate attributions
    attributions, delta = ig.attribute(input_seq, target=0, return_convergence_delta=True)
    
    # Sum attributions across the time dimension (30 windows) to get overall feature importance
    attr_sum = attributions.squeeze(0).sum(dim=0).detach().cpu().numpy()
    
    # Pair with feature names and sort by absolute importance
    importance = []
    for i, name in enumerate(feature_names):
        importance.append({"Feature": name, "Contribution": attr_sum[i]})
        
    # Sort descending by absolute contribution
    importance = sorted(importance, key=lambda x: abs(x["Contribution"]), reverse=True)
    
    return importance[:5] # Return top 5 features
