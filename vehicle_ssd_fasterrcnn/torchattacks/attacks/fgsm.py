import torch
import torch.nn as nn

from ..attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]

    # def forward(self, images, labels):
    def forward(self, images, targets):
        r"""
        Overridden.
        """
        images = torch.stack(images, dim=0)
        
        images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        # outputs = self.get_logits(images)
        images_list = [images[i] for i in range(images.shape[0])]
        self.model.train() # train才会求loss
        loss_dict = self.model(images_list, targets)
        
        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            # cost = loss(outputs, labels)
            cost = sum(loss for loss in loss_dict.values())

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        # return adv_images
        return [adv_images[i] for i in range(adv_images.shape[0])]
