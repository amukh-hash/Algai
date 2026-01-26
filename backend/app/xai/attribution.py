import torch

try:
    from captum.attr import IntegratedGradients
except ImportError:
    IntegratedGradients = None

class FeatureAttributor:
    def __init__(self, model):
        self.model = model
        if IntegratedGradients:
            self.ig = IntegratedGradients(self._forward_wrapper)
        else:
            self.ig = None
        
    def _forward_wrapper(self, x):
        # Integrated Gradients requires a scalar output or specific target index.
        output = self.model(x)
        # HybridPatchTST returns (direction, volatility)
        if isinstance(output, tuple):
            output = output[0] # Attribute 'direction'
        return output.sum()

    def attribute(self, x_input, n_steps=50):
        """
        Compute attributions.
        x_input: (Batch, Lookback, Features)
        """
        self.model.eval()
        
        if self.ig is None:
            # Return dummy attributions if captum is missing
            return torch.zeros_like(x_input)

        # Define baseline as zero
        baseline = torch.zeros_like(x_input)
        
        attributions = self.ig.attribute(x_input, baseline, n_steps=n_steps)
        return attributions
