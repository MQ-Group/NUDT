import torch
import torch.nn as nn

from torchattacks.attack import Attack

class YOPO(Attack):
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

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        loss_fn = nn.CrossEntropyLoss()
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Initialize random perturbation
        delta = torch.zeros_like(images).to(self.device)
        delta.uniform_(-self.eps, self.eps)
        delta = torch.clamp(delta, -self.eps, self.eps)
        delta.requires_grad = True
        
        # Only one forward and backward propagation as in YOPO
        outputs = self.get_logits(images + delta)
        
        loss = loss_fn(outputs, labels)
        grad = torch.autograd.grad(loss, delta)[0]
        
        # Update perturbation using the gradient
        for _ in range(self.steps):
            delta.data = delta.data + self.alpha * torch.sign(grad)
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images
        
        adv_images = (images + delta).detach()
        
        return adv_images

