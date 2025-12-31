import torch
import torch.nn as nn

from torchattacks.attack import Attack

class TRADES(Attack):
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

        adv_images = images.detach() + 0.001 * torch.randn(images.shape).to(self.device).detach()
        for _ in range(self.steps):
            adv_images.requires_grad_()
            with torch.enable_grad():
                # perturb based on logistic loss
                outputs = self.get_logits(adv_images)
                loss = loss_fn(outputs, labels)
                
            grad = torch.autograd.grad(loss, [adv_images])[0]
            adv_images = adv_images.detach() + self.alpha * torch.sign(grad.detach())
            adv_images = torch.min(torch.max(adv_images, images - self.eps), images + self.eps)
            adv_images = torch.clamp(adv_images, 0.0, 1.0)
        
        return adv_images

