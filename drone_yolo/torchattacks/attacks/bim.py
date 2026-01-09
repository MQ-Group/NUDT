import torch
import torch.nn as nn

from ..attack import Attack


class BIM(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps
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

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()

        # ori_images = images.clone().detach()
        ori_images = batch['img'].clone().detach()

        for _ in range(self.steps):
            # images.requires_grad = True
            # outputs = self.get_logits(images)
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
            #     cost, images, retain_graph=False, create_graph=False
            # )[0]
            grad = torch.autograd.grad(
                cost, batch['img'], retain_graph=False, create_graph=False
            )[0]

            # adv_images = images + self.alpha * grad.sign()
            adv_images = batch['img'] + self.alpha * grad.sign()
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images + (
                adv_images < a
            ).float() * a  # nopep8
            c = (b > ori_images + self.eps).float() * (ori_images + self.eps) + (
                b <= ori_images + self.eps
            ).float() * b  # nopep8
            # images = torch.clamp(c, max=1).detach()
            batch['img'] = torch.clamp(c, max=1).detach()

        # return images
        return batch['img']
