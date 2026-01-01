
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import random

from ..attack import Attack

class Boundary(Attack):
    r"""
    Boundary Attack in the paper 'Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models'
    [https://arxiv.org/abs/1712.04248]

    This is a decision-based black-box attack that only requires the final 
    classification decision (labels) of the model.

    Arguments:
        model (nn.Module): model to attack.
        max_queries (int): maximum number of queries to the model. (Default: 1000)
        init_epsilon (float): initial step size for random walk. (Default: 0.1)
        spherical_step (float): step size for spherical boundary step. (Default: 0.01)
        orthogonal_step (float): step size for orthogonal step. (Default: 0.01)
        binary_search_steps (int): number of binary search steps. (Default: 10)

    Shape:
        ▪ images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        ▪ labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        ▪ output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.Boundary(model, max_queries=1000)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, max_queries=1000, 
                 init_epsilon=0.1, spherical_step=0.01, 
                 orthogonal_step=0.01, binary_search_steps=10):
        super().__init__("Boundary", model)
        self.max_queries = max_queries
        self.init_epsilon = init_epsilon
        self.spherical_step = spherical_step
        self.orthogonal_step = orthogonal_step
        self.binary_search_steps = binary_search_steps
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
        Attack a single image
        """
        # Initialize adversarial image
        if self.targeted:
            # For targeted attack, we want to move towards the target class
            # Start with a random perturbation
            adv_image = image + torch.randn_like(image) * self.init_epsilon
            adv_image = torch.clamp(adv_image, 0, 1)
        else:
            # For untargeted attack, start with a random point that is already adversarial
            # We need to find an initial adversarial example
            adv_image = self._find_initial_adversarial(image, label)
        
        # Get original prediction
        with torch.no_grad():
            orig_output = self.get_logits(image)
            orig_pred = orig_output.argmax(1)
        
        queries = 0
        best_adv = adv_image.clone()
        best_distance = float('inf')
        
        # Main attack loop
        for _ in range(self.max_queries // 2):  # Each iteration uses 2 queries
            if queries >= self.max_queries:
                break
            
            # Step 1: Orthogonal step - move towards original image
            orth_step = self._orthogonal_step(image, adv_image)
            
            # Step 2: Spherical step - move along decision boundary
            new_adv = self._spherical_step(adv_image, orth_step)
            new_adv = torch.clamp(new_adv, 0, 1)
            
            # Check if new point is adversarial
            queries += 1
            with torch.no_grad():
                new_output = self.get_logits(new_adv)
                new_pred = new_output.argmax(1)
            
            is_adversarial = False
            if self.targeted:
                is_adversarial = (new_pred == target_label).item()
            else:
                is_adversarial = (new_pred != orig_pred).item()
            
            if is_adversarial:
                adv_image = new_adv
                
                # Check distance
                distance = torch.norm((adv_image - image).view(-1), p=2).item()
                if distance < best_distance:
                    best_distance = distance
                    best_adv = adv_image.clone()
            else:
                # Binary search to find boundary point
                adv_image = self._binary_search_boundary(image, adv_image, label, target_label)
            
            queries += 1
            
            # Early stopping if distance is very small
            if best_distance < 1e-4:
                break
        
        return best_adv
    
    def _find_initial_adversarial(self, image, label):
        """
        Find an initial adversarial example for untargeted attack
        """
        queries = 0
        max_initial_queries = min(100, self.max_queries // 10)
        
        for _ in range(max_initial_queries):
            # Generate random perturbation
            random_pert = torch.randn_like(image) * self.init_epsilon
            candidate = image + random_pert
            candidate = torch.clamp(candidate, 0, 1)
            
            queries += 1
            with torch.no_grad():
                output = self.get_logits(candidate)
                pred = output.argmax(1)
            
            if pred != label:
                return candidate
        
        # If no adversarial found, return a random point
        return image + torch.randn_like(image) * self.init_epsilon
    
    def _orthogonal_step(self, original, adversarial):
        """
        Take a step towards the original image
        """
        diff = original - adversarial
        diff_norm = torch.norm(diff.view(-1), p=2)
        
        if diff_norm < 1e-10:
            return adversarial
        
        step_size = self.orthogonal_step
        step = diff / diff_norm * step_size
        
        return adversarial + step
    
    def _spherical_step(self, adversarial, orthogonal_step_result):
        """
        Take a step along the decision boundary
        """
        # Generate random direction
        random_dir = torch.randn_like(adversarial)
        
        # Orthogonalize random direction with respect to (orthogonal_step_result - adversarial)
        diff = orthogonal_step_result - adversarial
        diff_flat = diff.view(-1)
        random_dir_flat = random_dir.view(-1)
        
        # Project random direction onto the orthogonal complement of diff
        proj = (torch.dot(random_dir_flat, diff_flat) / 
                torch.dot(diff_flat, diff_flat)) * diff_flat
        orth_random = random_dir_flat - proj
        
        # Normalize
        orth_random_norm = torch.norm(orth_random, p=2)
        if orth_random_norm < 1e-10:
            orth_random = torch.randn_like(orth_random)
            orth_random_norm = torch.norm(orth_random, p=2)
        
        orth_random = orth_random / orth_random_norm
        
        # Take step
        step = orth_random.view_as(adversarial) * self.spherical_step
        return orthogonal_step_result + step
    
    def _binary_search_boundary(self, original, adversarial, label, target_label=None):
        """
        Perform binary search to find the boundary point
        """
        low = original.clone()
        high = adversarial.clone()
        
        with torch.no_grad():
            orig_output = self.get_logits(original)
            orig_pred = orig_output.argmax(1)
        
        for _ in range(self.binary_search_steps):
            mid = (low + high) / 2
            mid = torch.clamp(mid, 0, 1)
            
            with torch.no_grad():
                mid_output = self.get_logits(mid)
                mid_pred = mid_output.argmax(1)
            
            is_adversarial = False
            if self.targeted:
                is_adversarial = (mid_pred == target_label).item()
            else:
                is_adversarial = (mid_pred != orig_pred).item()
            
            if is_adversarial:
                high = mid
            else:
                low = mid
        
        return high