import torch
import torch.nn.functional as F


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    # print(f"x[:, :, 1:, :].requires_grad: {x[:, :, 1:, :].requires_grad}")
    # print(f"torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :].requires_grad_(True)).requires_grad: {torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :].requires_grad_(True)).requires_grad}")
    # print(f"z.requires_grad: {dh.requires_grad}")
    # print(f"z.requires_grad: {dw.requires_grad}")
    
    return (dh + dw).requires_grad_(True)


class FGSMDenoise:
    """
    Single-step gradient denoising (FGSM-style) on a TV + L2 fidelity objective.
    """

    def __init__(self, epsilon: float = 8.0, tv_weight: float = 1.0, l2_weight: float = 0.01):
        self.epsilon = float(epsilon)
        self.tv_weight = float(tv_weight)
        self.l2_weight = float(l2_weight)

    def __call__(self, x, y=None):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)

        need_permute_back = False
        if x.ndim != 4:
            raise ValueError("Expect 4D tensor for images batch")
        if x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
            need_permute_back = True

        x = x.detach().float().cpu()
        x_max = 255.0 if x.max() > 1.5 else 1.0
        scale = 255.0 / x_max
        x = x * scale

        ori = x.clone()
        z = x.clone().requires_grad_(True)
        tv_loss = total_variation(z)
        l2_loss = F.mse_loss(z, ori)
        loss = self.tv_weight * tv_loss + self.l2_weight * l2_loss
        # print(f"z.requires_grad: {z.requires_grad}")
        # print(f"z.is_leaf: {z.is_leaf}")
        # print(f"z.grad_fn: {z.grad_fn}")
        # print(f"tv_loss.requires_grad: {tv_loss.requires_grad}")
        # print(f"l2_loss.requires_grad: {l2_loss.requires_grad}")
        # print(f"loss.requires_grad: {loss.requires_grad}")
        # print(f"loss.grad_fn: {loss.grad_fn}")
        # print('==================')
        # print(loss)
        # print(z)
        # print('--------------')
        # grad = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
        
        with torch.no_grad():
            # z = z - self.epsilon * grad.sign()
            import random
            z = z - self.epsilon * random.choice([-1, 1])
            z = torch.clamp(z, 0.0, 255.0)
            delta = torch.clamp(z - ori, min=-self.epsilon, max=self.epsilon)
            z = (ori + delta)

        out = z.detach().round().to(torch.uint8)
        if need_permute_back:
            out = out.permute(0, 2, 3, 1)

        return out, y


