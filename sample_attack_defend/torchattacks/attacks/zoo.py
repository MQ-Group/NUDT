import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
import random

from ..attack import Attack


class ZOO(Attack):
    r"""
    ZOO (Zeroth Order Optimization) in the paper 'ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models'
    [https://arxiv.org/abs/1708.03999]

    This is a score-based black-box attack that only requires the model's 
    probability outputs (scores) rather than gradients.

    Arguments:
        model (nn.Module): model to attack.
        max_iterations (int): maximum number of iterations. (Default: 100)
        learning_rate (float): learning rate for optimization. (Default: 0.01)
        binary_search_steps (int): number of binary search steps for const. (Default: 5)
        init_const (float): initial constant for balancing L2 distance and classification loss. (Default: 0.01)
        beta (float): beta parameter for L1 regularization. (Default: 0.001)
        batch_size (int): batch size for coordinate descent. (Default: 128)
        resolution (int): resolution for coordinate-wise update. (Default: 1)
        early_stop_iters (int): early stopping iterations. (Default: 10)
        abort_early (bool): flag for early abortion. (Default: True)

    Shape:
        ▪ images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        ▪ labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        ▪ output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.ZOO(model, max_iterations=100)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, max_iterations=100, 
                 learning_rate=0.01, binary_search_steps=5, init_const=0.01,
                 beta=0.001, batch_size=128, resolution=1, 
                 early_stop_iters=10, abort_early=True):
        super().__init__("ZOO", model)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.init_const = init_const
        self.beta = beta
        self.batch_size = batch_size
        self.resolution = resolution
        self.early_stop_iters = early_stop_iters
        self.abort_early = abort_early
        self.supported_mode = ["default", "targeted"]
        
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        
        adv_images = []
        for idx in range(len(images)):
            image = images[idx:idx+1]
            label = labels[idx:idx+1]
            
            if self.targeted:
                target_label = target_labels[idx:idx+1]
                adv_image = self._attack_single(image, label, target_label)
            else:
                adv_image = self._attack_single(image, label)
            
            adv_images.append(adv_image)
        
        adv_images = torch.cat(adv_images, dim=0)
        return adv_images
    
    def _attack_single(self, image, label, target_label=None):
        """
        Attack a single image using ZOO
        """
        # Get image dimensions
        n_channels = image.shape[1]
        height = image.shape[2]
        width = image.shape[3]
        
        # Reshape image to 1D
        image_flat = image.view(-1).cpu().numpy()
        n_pixels = len(image_flat)
        
        # Initialize adversarial image
        adv_flat = image_flat.copy()
        adv_image = image.clone()
        
        # Binary search for const parameter
        const = self.init_const
        lower_bound = 0.0
        upper_bound = 1e10
        
        # Variables for ADAM optimizer
        m = np.zeros_like(image_flat)  # first moment
        v = np.zeros_like(image_flat)  # second moment
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        best_adv = adv_flat.copy()
        best_l2 = float('inf')
        best_loss = float('inf')
        
        for binary_search_step in range(self.binary_search_steps):
            if binary_search_step > 0:
                const = (lower_bound + upper_bound) / 2
            
            # print(f"Binary search step {binary_search_step + 1}/{self.binary_search_steps}, const={const}")
            
            # Reset ADAM moments
            m.fill(0)
            v.fill(0)
            
            # Variables for early stopping
            last_losses = [1e10] * self.early_stop_iters
            current_best_l2 = float('inf')
            current_best_adv = adv_flat.copy()
            
            for iteration in range(self.max_iterations):
                # Generate random directions for coordinate descent
                indices = np.random.choice(n_pixels, min(self.batch_size, n_pixels), replace=False)
                
                # Estimate gradients using symmetric difference quotient
                grad_estimate = np.zeros_like(adv_flat)
                
                for idx in indices:
                    # Positive perturbation
                    adv_pos = adv_flat.copy()
                    adv_pos[idx] += self.resolution
                    adv_pos_img = torch.from_numpy(adv_pos.reshape(image.shape)).float().to(self.device)
                    
                    # Negative perturbation
                    adv_neg = adv_flat.copy()
                    adv_neg[idx] -= self.resolution
                    adv_neg_img = torch.from_numpy(adv_neg.reshape(image.shape)).float().to(self.device)
                    
                    # Get model outputs
                    with torch.no_grad():
                        output_pos = self.get_logits(adv_pos_img)
                        output_neg = self.get_logits(adv_neg_img)
                    
                    # Calculate loss
                    loss_pos = self._loss_function(adv_pos_img, image, output_pos, label, target_label, const)
                    loss_neg = self._loss_function(adv_neg_img, image, output_neg, label, target_label, const)
                    
                    # Finite difference gradient estimate
                    grad_estimate[idx] = (loss_pos - loss_neg) / (2 * self.resolution)
                
                # ADAM update
                t = iteration + 1
                m = beta1 * m + (1 - beta1) * grad_estimate
                v = beta2 * v + (1 - beta2) * (grad_estimate ** 2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                
                # Update adversarial example
                adv_flat -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
                
                # Project back to [0, 1] range
                adv_flat = np.clip(adv_flat, 0, 1)
                
                # Reshape to image
                adv_image_np = adv_flat.reshape(image.shape)
                adv_image = torch.from_numpy(adv_image_np).float().to(self.device)
                
                # Calculate current loss
                with torch.no_grad():
                    output = self.get_logits(adv_image)
                
                current_loss = self._loss_function(adv_image, image, output, label, target_label, const)
                current_l2 = torch.norm((adv_image - image).view(-1), p=2).item()
                
                # Check if attack is successful
                current_pred = output.argmax(1).item()
                is_successful = False
                
                if self.targeted:
                    is_successful = (current_pred == target_label.item())
                else:
                    is_successful = (current_pred != label.item())
                
                # Update best solution
                if is_successful and current_l2 < best_l2:
                    best_l2 = current_l2
                    best_adv = adv_flat.copy()
                
                if is_successful and current_l2 < current_best_l2:
                    current_best_l2 = current_l2
                    current_best_adv = adv_flat.copy()
                
                # Early stopping check
                # if iteration % 10 == 0:
                #     print(f"Iteration {iteration + 1}/{self.max_iterations}, Loss: {current_loss:.4f}, L2: {current_l2:.4f}, Success: {is_successful}")
                
                if self.abort_early and iteration % self.early_stop_iters == 0:
                    if current_loss > last_losses[0] * 0.9999:
                        # print(f"Early stopping at iteration {iteration + 1}")
                        break
                    last_losses = last_losses[1:] + [current_loss]
            
            # Update binary search bounds
            adv_image_tensor = torch.from_numpy(current_best_adv.reshape(image.shape)).float().to(self.device)
            with torch.no_grad():
                output = self.get_logits(adv_image_tensor)
            current_pred = output.argmax(1).item()
            
            if self.targeted:
                is_successful = (current_pred == target_label.item())
            else:
                is_successful = (current_pred != label.item())
            
            if is_successful:
                # Success, try smaller const
                upper_bound = const
            else:
                # Failure, try larger const
                lower_bound = const
            
            if upper_bound == 1e10:
                const *= 10
            else:
                const = (lower_bound + upper_bound) / 2
        
        # Return best adversarial example
        best_adv_img = torch.from_numpy(best_adv.reshape(image.shape)).float().to(self.device)
        return best_adv_img
    
    def _loss_function(self, adv_image, orig_image, output, label, target_label=None, const=0.01):
        """
        Calculate loss function: L = L2_distance + const * classification_loss
        """
        # L2 distance
        l2_dist = torch.sum((adv_image - orig_image) ** 2)
        
        # Classification loss
        if self.targeted:
            # Targeted attack: maximize probability of target class
            target_probs = torch.softmax(output, dim=1)[0, target_label]
            class_loss = torch.max(torch.tensor(0.0).to(self.device), 1.0 - target_probs)
        else:
            # Untargeted attack: minimize probability of true class
            true_probs = torch.softmax(output, dim=1)[0, label]
            class_loss = torch.max(torch.tensor(0.0).to(self.device), true_probs - 0.5)
        
        # Total loss
        total_loss = l2_dist + const * class_loss
        
        return total_loss.item()
    
    def _estimate_gradient(self, image_flat, orig_image, label, target_label=None, indices=None, h=0.0001):
        """
        Estimate gradient using symmetric difference quotient
        """
        n_pixels = len(image_flat)
        if indices is None:
            indices = np.random.choice(n_pixels, min(self.batch_size, n_pixels), replace=False)
        
        grad_estimate = np.zeros_like(image_flat)
        
        for idx in indices:
            # Positive perturbation
            image_pos = image_flat.copy()
            image_pos[idx] += h
            image_pos_tensor = torch.from_numpy(image_pos.reshape(orig_image.shape)).float().to(self.device)
            
            # Negative perturbation
            image_neg = image_flat.copy()
            image_neg[idx] -= h
            image_neg_tensor = torch.from_numpy(image_neg.reshape(orig_image.shape)).float().to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                output_pos = self.get_logits(image_pos_tensor)
                output_neg = self.get_logits(image_neg_tensor)
            
            # Calculate losses
            loss_pos = self._loss_function(image_pos_tensor, orig_image, output_pos, label, target_label)
            loss_neg = self._loss_function(image_neg_tensor, orig_image, output_neg, label, target_label)
            
            # Finite difference gradient
            grad_estimate[idx] = (loss_pos - loss_neg) / (2 * h)
        
        return grad_estimate