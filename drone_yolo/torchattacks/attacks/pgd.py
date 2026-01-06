import torch
import torch.nn as nn

from ..attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    # def forward(self, images, labels):
    def forward(self, batch):
        r"""
        Overridden.
        """

        # images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)
        batch['img'] = batch['img'].clone().detach().to(self.device)
        batch['cls'] = batch['cls'].clone().detach().to(self.device)
        batch['bboxes'] = batch['bboxes'].clone().detach().to(self.device)
        
        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()
        # adv_images = images.clone().detach()
        adv_images = batch['img'].clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        images = batch['img']
        batch['img'] = adv_images
        
        for _ in range(self.steps):
            # adv_images.requires_grad = True
            # outputs = self.get_logits(adv_images)
            batch['img'].requires_grad = True
            loss, loss_items = self.model(batch)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                # cost = loss(outputs, labels)
                cost = loss.sum()

            # Update adversarial images
            # grad = torch.autograd.grad(
            #     cost, adv_images, retain_graph=False, create_graph=False
            # )[0]
            grad = torch.autograd.grad(
                cost, batch['img'], retain_graph=False, create_graph=False
            )[0]

            # adv_images = adv_images.detach() + self.alpha * grad.sign()
            # delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            # adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            batch['img'] = batch['img'].detach() + self.alpha * grad.sign()
            delta = torch.clamp(batch['img'] - images, min=-self.eps, max=self.eps)
            batch['img'] = torch.clamp(images + delta, min=0, max=1).detach()

        # return adv_images
        return batch['img']
