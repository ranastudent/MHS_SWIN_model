import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Forward hook
        target_layer.register_forward_hook(self.save_activation)

        # Backward hook
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx=None):
        self.model.eval()

        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()

        grads = self.gradients
        acts = self.activations

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * acts, dim=1)
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam