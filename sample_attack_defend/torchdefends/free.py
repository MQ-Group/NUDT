import torch
import torch.nn as nn

from torchattacks.attack import Attack

class FREE(Attack):
    r"""
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchdefend.YOPO(model, eps=8/255, alpha=1/255, steps=10)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10):
        super().__init__("YOPO", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ["default", "targeted"]
        self.restarts = 1  # number of random restarts
        
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        loss_fn = nn.CrossEntropyLoss()
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Initialize perturbation
        delta = torch.zeros_like(images).to(self.device)
        delta.requires_grad = True

        # Multi-restart for better adversarial examples
        for _ in range(self.restarts):
            # Reset gradients
            if delta.grad is not None:
                delta.grad.zero_()
            
            # Generate adversarial examples
            for _ in range(self.steps):
                outputs = self.get_logits(images + delta)
                loss = loss_fn(outputs, labels)
                
                # Backward pass to compute gradients w.r.t. delta
                loss.backward()
                
                # Update perturbation using FGSM
                delta.data = delta.data + self.alpha * delta.grad.sign()
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)
                delta.data = torch.clamp(images + delta.data, 0, 1) - images
                
                # Zero gradients for next iteration
                delta.grad.zero_()
        
        adv_images = (images + delta).detach()
        
        return adv_images
