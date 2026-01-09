import torch
import torch.nn as nn
import numpy as np
import math
from scipy.spatial.distance import cdist
import random

from ..attack import Attack


class HSJA(Attack):
    r"""
    HSJA (HopSkipJumpAttack) in the paper 'HopSkipJumpAttack: A Query-Efficient Decision-Based Attack'
    [https://arxiv.org/abs/1904.02144]

    This is a decision-based black-box attack that only requires the final 
    classification decision (labels) of the model. It combines gradient estimation
    with binary search for efficient boundary exploration.

    Arguments:
        model (nn.Module): model to attack.
        max_queries (int): maximum number of queries to the model. (Default: 1000)
        norm (str): norm to measure distance. Supports 'L2' or 'Linf'. (Default: 'L2')
        gamma (float): parameter to control the step size. (Default: 0.01)
        init_num_evals (int): initial number of evaluations for gradient estimation. (Default: 100)
        max_num_evals (int): maximum number of evaluations for gradient estimation. (Default: 10000)
        stepsize_search (str): method for step size search. Supports 'geometric_progression' or 'grid_search'. (Default: 'geometric_progression')
        num_iterations (int): maximum number of iterations. (Default: 64)
        constraint (str): constraint for perturbation. Supports 'L2' or 'Linf'. (Default: 'L2')
        batch_size (int): batch size for gradient estimation. (Default: 128)

    Shape:
        ▪ images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        ▪ labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        ▪ output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.HSJA(model, max_queries=1000, norm='L2')
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, max_queries=1000, norm='L2',
                 gamma=0.01, init_num_evals=100, max_num_evals=10000,
                 stepsize_search='geometric_progression', num_iterations=64,
                 constraint='L2', batch_size=128):
        super().__init__("HSJA", model)
        self.max_queries = max_queries
        self.norm = norm
        self.gamma = gamma
        self.init_num_evals = init_num_evals
        self.max_num_evals = max_num_evals
        self.stepsize_search = stepsize_search
        self.num_iterations = num_iterations
        self.constraint = constraint
        self.batch_size = batch_size
        self.supported_mode = ["default", "targeted"]
        
        if norm not in ['L2', 'Linf']:
            raise ValueError("norm must be 'L2' or 'Linf'")
        if constraint not in ['L2', 'Linf']:
            raise ValueError("constraint must be 'L2' or 'Linf'")
        if stepsize_search not in ['geometric_progression', 'grid_search']:
            raise ValueError("stepsize_search must be 'geometric_progression' or 'grid_search'")
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        
        adv_images = []
        queries_used = []
        
        for idx in range(len(images)):
            image = images[idx:idx+1]
            label = labels[idx:idx+1]
            
            if self.targeted:
                target_label = target_labels[idx:idx+1]
                adv_image, queries = self._attack_single(image, label, target_label)
            else:
                adv_image, queries = self._attack_single(image, label)
            
            adv_images.append(adv_image)
            queries_used.append(queries)
        
        adv_images = torch.cat(adv_images, dim=0)
        # print(f"HSJA attack completed. Average queries: {np.mean(queries_used):.1f}")
        return adv_images
    
    def _attack_single(self, image, label, target_label=None):
        """
        Attack a single image using HSJA
        """
        # Initialize adversarial example
        if self.targeted:
            # For targeted attack, find an initial adversarial example
            x_adv = self._initialize_targeted(image, label, target_label)
        else:
            # For untargeted attack, find an initial adversarial example
            x_adv = self._initialize_untargeted(image, label)
        
        queries = 0
        original_shape = image.shape
        
        # Get original prediction
        with torch.no_grad():
            orig_output = self.get_logits(image)
            orig_pred = orig_output.argmax(1)
            queries += 1
        
        # Main iteration loop
        for iteration in range(self.num_iterations):
            if queries >= self.max_queries:
                break
            
            # Step 1: Binary search to find boundary point
            x_adv, boundary_queries = self._binary_search_boundary(image, x_adv, label, target_label)
            queries += boundary_queries
            
            if queries >= self.max_queries:
                break
            
            # Step 2: Estimate gradient at boundary
            gradient, grad_queries = self._estimate_gradient(image, x_adv, label, target_label, queries)
            queries += grad_queries
            
            if queries >= self.max_queries:
                break
            
            # Step 3: Determine step size
            if iteration == 0:
                if self.stepsize_search == 'geometric_progression':
                    # Find step size via geometric progression
                    epsilon = self._geometric_progression_for_stepsize(image, x_adv, gradient, label, target_label)
                else:
                    # Find step size via grid search
                    epsilon = self._grid_search_for_stepsize(image, x_adv, gradient, label, target_label)
            else:
                epsilon = self._select_stepsize(iteration, gradient)
            
            # Step 4: Update adversarial example
            if self.norm == 'L2':
                # L2 norm update
                x_adv = x_adv + epsilon * gradient / (torch.norm(gradient.view(-1), p=2) + 1e-10)
            else:
                # Linf norm update
                x_adv = x_adv + epsilon * torch.sign(gradient)
            
            # Project to valid range
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # Project to constraint set
            if self.constraint == 'L2':
                # L2 constraint: ensure perturbation is within a sphere
                perturbation = x_adv - image
                norm = torch.norm(perturbation.view(-1), p=2)
                if norm > self.gamma:
                    perturbation = perturbation / norm * self.gamma
                    x_adv = image + perturbation
            else:
                # Linf constraint: ensure perturbation is within a cube
                perturbation = x_adv - image
                perturbation = torch.clamp(perturbation, -self.gamma, self.gamma)
                x_adv = image + perturbation
            
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # Check if adversarial
            with torch.no_grad():
                output = self.get_logits(x_adv)
                pred = output.argmax(1)
                queries += 1
            
            is_adversarial = False
            if self.targeted:
                is_adversarial = (pred == target_label).item()
            else:
                is_adversarial = (pred != orig_pred).item()
            
            if not is_adversarial:
                # If not adversarial, project back to boundary
                x_adv, boundary_queries = self._binary_search_boundary(image, x_adv, label, target_label)
                queries += boundary_queries
            
            if iteration % 10 == 0:
                dist = torch.norm((x_adv - image).view(-1), p=2).item()
                print(f"Iteration {iteration + 1}/{self.num_iterations}, Queries: {queries}, Distance: {dist:.6f}, Adversarial: {is_adversarial}")
            
            if queries >= self.max_queries:
                break
        
        return x_adv, queries
    
    def _initialize_untargeted(self, image, label):
        """
        Initialize adversarial example for untargeted attack
        """
        # Start with a random perturbation
        for _ in range(100):  # Try 100 random perturbations
            if self.norm == 'L2':
                # Random direction with L2 norm
                random_dir = torch.randn_like(image)
                random_dir = random_dir / torch.norm(random_dir.view(-1), p=2)
                x_adv = image + self.gamma * random_dir
            else:
                # Random direction with Linf norm
                random_dir = torch.rand_like(image) * 2 - 1  # Uniform in [-1, 1]
                x_adv = image + self.gamma * random_dir
            
            x_adv = torch.clamp(x_adv, 0, 1)
            
            with torch.no_grad():
                output = self.get_logits(x_adv)
                pred = output.argmax(1)
            
            if pred != label:
                return x_adv
        
        # If no adversarial found, return the last one
        return x_adv
    
    def _initialize_targeted(self, image, label, target_label):
        """
        Initialize adversarial example for targeted attack
        """
        # Start with a random perturbation
        for _ in range(100):  # Try 100 random perturbations
            if self.norm == 'L2':
                # Random direction with L2 norm
                random_dir = torch.randn_like(image)
                random_dir = random_dir / torch.norm(random_dir.view(-1), p=2)
                x_adv = image + self.gamma * random_dir
            else:
                # Random direction with Linf norm
                random_dir = torch.rand_like(image) * 2 - 1  # Uniform in [-1, 1]
                x_adv = image + self.gamma * random_dir
            
            x_adv = torch.clamp(x_adv, 0, 1)
            
            with torch.no_grad():
                output = self.get_logits(x_adv)
                pred = output.argmax(1)
            
            if pred == target_label:
                return x_adv
        
        # If no adversarial found, return the last one
        return x_adv
    
    def _binary_search_boundary(self, image, x_adv, label, target_label=None):
        """
        Perform binary search to find the boundary point
        """
        queries = 0
        
        with torch.no_grad():
            orig_output = self.get_logits(image)
            orig_pred = orig_output.argmax(1)
            queries += 1
        
        # Check if x_adv is adversarial
        with torch.no_grad():
            adv_output = self.get_logits(x_adv)
            adv_pred = adv_output.argmax(1)
            queries += 1
        
        is_adversarial = False
        if self.targeted:
            is_adversarial = (adv_pred == target_label).item()
        else:
            is_adversarial = (adv_pred != orig_pred).item()
        
        if not is_adversarial:
            # x_adv is not adversarial, return original image
            return image, queries
        
        # Binary search between image and x_adv
        low = torch.zeros_like(image)
        high = torch.ones_like(x_adv)
        
        # Ensure high is adversarial
        high.data.copy_(x_adv)
        
        for _ in range(20):  # Max 20 binary search steps
            mid = (low + high) / 2
            mid = torch.clamp(mid, 0, 1)
            
            with torch.no_grad():
                mid_output = self.get_logits(mid)
                mid_pred = mid_output.argmax(1)
                queries += 1
            
            is_mid_adversarial = False
            if self.targeted:
                is_mid_adversarial = (mid_pred == target_label).item()
            else:
                is_mid_adversarial = (mid_pred != orig_pred).item()
            
            if is_mid_adversarial:
                high.data.copy_(mid)
            else:
                low.data.copy_(mid)
        
        return high, queries
    
    def _estimate_gradient(self, image, x_adv, label, target_label=None, queries_used=0):
        """
        Estimate gradient at the boundary point using Monte Carlo sampling
        """
        queries = 0
        n_samples = min(self.init_num_evals, self.max_queries - queries_used)
        
        if n_samples <= 0:
            # Return zero gradient if no queries left
            return torch.zeros_like(x_adv), queries
        
        # Generate random directions
        if self.norm == 'L2':
            # Sample from unit sphere
            random_dirs = torch.randn(n_samples, *x_adv.shape[1:], device=self.device)
            norms = torch.norm(random_dirs.view(n_samples, -1), p=2, dim=1, keepdim=True)
            random_dirs = random_dirs / norms.view(n_samples, 1, 1, 1)
        else:
            # Sample from unit cube
            random_dirs = torch.rand(n_samples, *x_adv.shape[1:], device=self.device) * 2 - 1
            random_dirs = random_dirs / torch.norm(random_dirs.view(n_samples, -1), p=float('inf'), dim=1, keepdim=True).view(n_samples, 1, 1, 1)
        
        # Perturb x_adv in random directions
        delta = 0.01  # Small perturbation
        grad_estimates = []
        
        for i in range(0, n_samples, self.batch_size):
            batch_dirs = random_dirs[i:i+self.batch_size]
            batch_size = batch_dirs.shape[0]
            
            # Create perturbed images
            perturbed = x_adv.repeat(batch_size, 1, 1, 1) + delta * batch_dirs
            perturbed = torch.clamp(perturbed, 0, 1)
            
            # Query model
            with torch.no_grad():
                outputs = self.get_logits(perturbed)
                preds = outputs.argmax(1)
                queries += batch_size
            
            # Determine if perturbed points are adversarial
            is_adversarial = []
            for j in range(batch_size):
                if self.targeted:
                    is_adv = (preds[j] == target_label).item()
                else:
                    is_adv = (preds[j] != label).item()
                is_adversarial.append(is_adv)
            
            # Compute gradient estimate
            for j in range(batch_size):
                if is_adversarial[j]:
                    grad_estimates.append(batch_dirs[j])
                else:
                    grad_estimates.append(-batch_dirs[j])
        
        # Average gradient estimates
        if len(grad_estimates) > 0:
            grad_estimate = torch.stack(grad_estimates).mean(dim=0, keepdim=True)
        else:
            grad_estimate = torch.zeros_like(x_adv)
        
        # Normalize gradient
        if self.norm == 'L2':
            grad_norm = torch.norm(grad_estimate.view(-1), p=2)
            if grad_norm > 0:
                grad_estimate = grad_estimate / grad_norm
        else:
            grad_norm = torch.norm(grad_estimate.view(-1), p=float('inf'))
            if grad_norm > 0:
                grad_estimate = grad_estimate / grad_norm
        
        return grad_estimate, queries
    
    def _geometric_progression_for_stepsize(self, image, x_adv, gradient, label, target_label=None):
        """
        Find step size via geometric progression
        """
        epsilon = 1.0
        queries = 0
        
        with torch.no_grad():
            orig_output = self.get_logits(image)
            orig_pred = orig_output.argmax(1)
            queries += 1
        
        # Try increasing step sizes
        for _ in range(20):  # Max 20 steps
            candidate = x_adv + epsilon * gradient
            candidate = torch.clamp(candidate, 0, 1)
            
            with torch.no_grad():
                output = self.get_logits(candidate)
                pred = output.argmax(1)
                queries += 1
            
            is_adversarial = False
            if self.targeted:
                is_adversarial = (pred == target_label).item()
            else:
                is_adversarial = (pred != orig_pred).item()
            
            if is_adversarial:
                break
            
            epsilon *= 0.5
        
        return epsilon
    
    def _grid_search_for_stepsize(self, image, x_adv, gradient, label, target_label=None):
        """
        Find step size via grid search
        """
        best_epsilon = 0
        best_distance = float('inf')
        queries = 0
        
        with torch.no_grad():
            orig_output = self.get_logits(image)
            orig_pred = orig_output.argmax(1)
            queries += 1
        
        # Search over a grid of epsilon values
        epsilon_grid = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        for epsilon in epsilon_grid:
            candidate = x_adv + epsilon * gradient
            candidate = torch.clamp(candidate, 0, 1)
            
            with torch.no_grad():
                output = self.get_logits(candidate)
                pred = output.argmax(1)
                queries += 1
            
            is_adversarial = False
            if self.targeted:
                is_adversarial = (pred == target_label).item()
            else:
                is_adversarial = (pred != orig_pred).item()
            
            if is_adversarial:
                distance = torch.norm((candidate - image).view(-1), p=2).item()
                if distance < best_distance:
                    best_distance = distance
                    best_epsilon = epsilon
        
        if best_epsilon == 0:
            # If no epsilon found, use a small value
            best_epsilon = 0.001
        
        return best_epsilon
    
    def _select_stepsize(self, iteration, gradient):
        """
        Select step size based on iteration
        """
        # Adaptive step size scheduling
        if iteration < 10:
            return 1.0
        elif iteration < 20:
            return 0.5
        elif iteration < 30:
            return 0.1
        else:
            return 0.01