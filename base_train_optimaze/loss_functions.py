import torch.nn.functional as F


def get_loss_fn(loss_fn_name):
    if loss_fn_name == 'cross_entropy':
        loss_function = F.cross_entropy
    elif loss_fn_name == 'mse':
        loss_function = F.mse_loss
    elif loss_fn_name == 'l1':
        loss_function = F.l1_loss
    elif loss_fn_name == 'binary_cross_entropy':
        loss_function = F.binary_cross_entropy
    else:
        raise ValueError("Invalid Loss Type!")
    return loss_function