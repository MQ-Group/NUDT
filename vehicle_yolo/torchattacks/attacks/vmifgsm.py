import torch
import torch.nn as nn

from ..attack import Attack


class VMIFGSM(Attack):
    r"""
    VMI-FGSM in the paper 'Enhancing the Transferability of Adversarial Attacks through Variance Tuning
    [https://arxiv.org/abs/2103.15571], Published as a conference paper at CVPR 2021
    Modified from "https://github.com/JHL-HUST/VT"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of iterations. (Default: 10)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        N (int): the number of sampled examples in the neighborhood. (Default: 5)
        beta (float): the upper bound of neighborhood. (Default: 3/2)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.VMIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, N=5, beta=3 / 2
    ):
        super().__init__("VMIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
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

        # momentum = torch.zeros_like(images).detach().to(self.device)
        # v = torch.zeros_like(images).detach().to(self.device)
        # loss = nn.CrossEntropyLoss()
        # adv_images = images.clone().detach()
        momentum = torch.zeros_like(batch['img']).detach().to(self.device)
        v = torch.zeros_like(batch['img']).detach().to(self.device)
        # loss = nn.CrossEntropyLoss()
        images = batch['img']
        adv_images = images.clone().detach()
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
            # adv_grad = torch.autograd.grad(
            #     cost, adv_images, retain_graph=False, create_graph=False
            # )[0]
            adv_grad = torch.autograd.grad(
                cost, batch['img'], retain_graph=False, create_graph=False
            )[0]

            grad = (adv_grad + v) / torch.mean(
                torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
            )
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = batch['img']

            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                # neighbor_images = adv_images.detach() + torch.randn_like(
                #     images
                # ).uniform_(-self.eps * self.beta, self.eps * self.beta)
                # neighbor_images.requires_grad = True
                # outputs = self.get_logits(neighbor_images)
                batch['img'] = adv_images.detach() + torch.randn_like(
                    images
                ).uniform_(-self.eps * self.beta, self.eps * self.beta)
                batch['img'].requires_grad = True
                loss, loss_items = self.model(batch)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    # cost = loss(outputs, labels)
                    cost = loss.sum()
                    
                # GV_grad += torch.autograd.grad(
                #     cost, neighbor_images, retain_graph=False, create_graph=False
                # )[0]
                GV_grad += torch.autograd.grad(
                    cost, batch['img'], retain_graph=False, create_graph=False
                )[0]
            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad

            batch['img'] = adv_images
            
            # adv_images = adv_images.detach() + self.alpha * grad.sign()
            # delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            # adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            batch['img'] = batch['img'].detach() + self.alpha * grad.sign()
            delta = torch.clamp(batch['img'] - images, min=-self.eps, max=self.eps)
            batch['img'] = torch.clamp(images + delta, min=0, max=1).detach()

        # return adv_images
        return batch['img']
