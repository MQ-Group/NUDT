import torch
import torch.nn as nn
import numpy as np
import random
import math

from ..attack import Attack


class NES(Attack):
    r"""
    NES (Natural Evolutionary Strategies) in the paper 'Black-box Adversarial Attacks with Limited Queries and Information'
    [https://arxiv.org/abs/1804.08598]

    This is a score-based black-box attack that uses Natural Evolutionary Strategies
    to estimate the gradient of the loss function with respect to the input.

    Arguments:
        model (nn.Module): model to attack.
        max_queries (int): maximum number of queries to the model. (Default: 1000)
        epsilon (float): maximum perturbation. (Default: 8/255)
        learning_rate (float): learning rate for gradient update. (Default: 0.01)
        samples_per_draw (int): number of samples per gradient estimation. (Default: 100)
        sigma (float): sampling variance. (Default: 0.001)
        decay_factor (float): decay factor for momentum. (Default: 0.9)
        norm (str): norm to measure distance. Supports 'L2' or 'Linf'. (Default: 'Linf')
        early_stop (bool): flag for early stopping. (Default: True)
        loss_func (str): loss function to use. Supports 'cross_entropy' or 'margin'. (Default: 'cross_entropy')

    Shape:
        ▪ images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        ▪ labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        ▪ output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.NES(model, max_queries=1000, epsilon=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, max_queries=1000, epsilon=8/255,
                 learning_rate=0.01, samples_per_draw=100, sigma=0.001,
                 decay_factor=0.9, norm='Linf', early_stop=True, loss_func='cross_entropy'):
        super().__init__("NES", model)
        self.max_queries = max_queries
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.samples_per_draw = samples_per_draw
        self.sigma = sigma
        self.decay_factor = decay_factor
        self.norm = norm
        self.early_stop = early_stop
        self.loss_func = loss_func
        self.supported_mode = ["default", "targeted"]
        
        if norm not in ['L2', 'Linf']:
            raise ValueError("norm must be 'L2' or 'Linf'")
        if loss_func not in ['cross_entropy', 'margin']:
            raise ValueError("loss_func must be 'cross_entropy' or 'margin'")
    
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
        print(f"NES attack completed. Average queries: {np.mean(queries_used):.1f}")
        return adv_images
    
    def _attack_single(self, image, label, target_label=None):
        """
        Attack a single image using NES
        """
        queries = 0
        original = image.clone()
        adv_image = image.clone()
        
        # Get original prediction
        with torch.no_grad():
            orig_output = self.get_logits(original)
            orig_pred = orig_output.argmax(1)
            queries += 1
        
        # Initialize momentum
        momentum = torch.zeros_like(image)
        
        # Early stopping variables
        best_adv = adv_image.clone()
        best_distance = float('inf')
        
        # Track if we've found an adversarial example
        found_adversarial = False
        
        for iteration in range(self.max_queries // self.samples_per_draw):
            if queries >= self.max_queries:
                break
            
            # Estimate gradient using NES
            gradient, batch_queries = self._estimate_gradient_nes(adv_image, label, target_label)
            queries += batch_queries
            
            # Update momentum
            momentum = self.decay_factor * momentum + gradient
            
            # Update adversarial image with momentum
            if self.norm == 'Linf':
                # Linf norm update
                adv_image = adv_image - self.learning_rate * torch.sign(momentum)
            else:
                # L2 norm update
                grad_norm = torch.norm(momentum.view(-1), p=2)
                if grad_norm > 0:
                    adv_image = adv_image - self.learning_rate * momentum / grad_norm
                else:
                    adv_image = adv_image - self.learning_rate * momentum
            
            # Project to epsilon ball
            adv_image = self._project_onto_epsilon_ball(original, adv_image)
            
            # Project to [0, 1] range
            adv_image = torch.clamp(adv_image, 0, 1)
            
            # Check if current point is adversarial
            with torch.no_grad():
                current_output = self.get_logits(adv_image)
                current_pred = current_output.argmax(1)
                queries += 1
            
            is_adversarial = False
            if self.targeted:
                is_adversarial = (current_pred == target_label).item()
            else:
                is_adversarial = (current_pred != orig_pred).item()
            
            # Calculate distance
            if self.norm == 'L2':
                distance = torch.norm((adv_image - original).view(-1), p=2).item()
            else:
                distance = torch.norm((adv_image - original).view(-1), p=float('inf')).item()
            
            # Update best solution
            if is_adversarial and distance < best_distance:
                best_distance = distance
                best_adv = adv_image.clone()
                found_adversarial = True
            
            # Early stopping if we have a good enough adversarial example
            if self.early_stop and found_adversarial and distance < self.epsilon * 0.9:
                print(f"Early stopping at iteration {iteration + 1}, distance: {distance:.6f}")
                break
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration + 1}, Queries: {queries}, Distance: {distance:.6f}, Adversarial: {is_adversarial}")
            
            if queries >= self.max_queries:
                break
        
        # If we never found an adversarial example, return the last one
        if not found_adversarial:
            best_adv = adv_image.clone()
        
        return best_adv, queries
    
    def _estimate_gradient_nes(self, image, label, target_label=None):
        """
        Estimate gradient using Natural Evolutionary Strategies
        """
        queries = 0
        n_samples = self.samples_per_draw
        
        # Sample perturbations from Gaussian distribution
        batch_size = min(n_samples, 100)  # Process in batches to avoid memory issues
        num_batches = math.ceil(n_samples / batch_size)
        
        # Initialize gradient estimate
        gradient_estimate = torch.zeros_like(image)
        total_weight = 0
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
            if current_batch_size <= 0:
                break
            
            # Sample noise
            noise = torch.randn(current_batch_size, *image.shape[1:], device=self.device)
            
            # Create positive and negative perturbations
            image_pos = image + self.sigma * noise
            image_neg = image - self.sigma * noise
            
            # Project to valid range
            image_pos = torch.clamp(image_pos, 0, 1)
            image_neg = torch.clamp(image_neg, 0, 1)
            
            # Get model outputs
            with torch.no_grad():
                # Combine positive and negative samples
                combined_images = torch.cat([image_pos, image_neg], dim=0)
                combined_outputs = self.get_logits(combined_images)
                queries += combined_images.shape[0]
            
            # Split outputs
            outputs_pos = combined_outputs[:current_batch_size]
            outputs_neg = combined_outputs[current_batch_size:]
            
            # Calculate losses
            losses_pos = self._calculate_loss(outputs_pos, label, target_label)
            losses_neg = self._calculate_loss(outputs_neg, label, target_label)
            
            # Compute gradient estimate for this batch
            for i in range(current_batch_size):
                weight = (losses_pos[i] - losses_neg[i])  # Difference in losses
                gradient_estimate += weight * noise[i:i+1]
                total_weight += abs(weight)
        
        # Normalize gradient estimate
        if total_weight > 0:
            gradient_estimate = gradient_estimate / (2 * n_samples * self.sigma * total_weight)
        else:
            gradient_estimate = gradient_estimate / (2 * n_samples * self.sigma + 1e-10)
        
        return gradient_estimate, queries
    
    def _calculate_loss(self, outputs, label, target_label=None):
        """
        Calculate loss for NES gradient estimation
        """
        if self.loss_func == 'cross_entropy':
            if self.targeted:
                # Targeted attack: minimize cross-entropy with target class
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                return loss_fn(outputs, target_label.repeat(outputs.shape[0]))
            else:
                # Untargeted attack: maximize cross-entropy with true class
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                return -loss_fn(outputs, label.repeat(outputs.shape[0]))
        else:  # margin loss
            if self.targeted:
                # Targeted attack: minimize margin between target and other classes
                target_probs = torch.softmax(outputs, dim=1)
                target_class_probs = target_probs[:, target_label]
                
                # Get second highest probability
                target_probs[:, target_label] = -float('inf')
                second_best_probs = torch.max(target_probs, dim=1)[0]
                
                return -(target_class_probs - second_best_probs)
            else:
                # Untargeted attack: maximize margin between true and other classes
                probs = torch.softmax(outputs, dim=1)
                true_class_probs = probs[:, label]
                
                # Get second highest probability
                probs[:, label] = -float('inf')
                second_best_probs = torch.max(probs, dim=1)[0]
                
                return true_class_probs - second_best_probs
    
    def _project_onto_epsilon_ball(self, original, perturbed):
        """
        Project perturbed image onto epsilon ball around original image
        """
        perturbation = perturbed - original
        
        if self.norm == 'Linf':
            # Linf norm projection
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        else:  # L2 norm
            # L2 norm projection
            norm = torch.norm(perturbation.view(-1), p=2)
            if norm > self.epsilon:
                perturbation = perturbation / norm * self.epsilon
        
        return original + perturbation
    
    def _initialize_adversarial(self, image, label, target_label=None):
        """
        Initialize adversarial example
        """
        # Start with original image
        adv_image = image.clone()
        
        # Add small random perturbation
        if self.norm == 'Linf':
            random_pert = (torch.rand_like(image) * 2 - 1) * self.epsilon * 0.1
        else:
            random_dir = torch.randn_like(image)
            random_norm = torch.norm(random_dir.view(-1), p=2)
            random_dir = random_dir / random_norm
            random_pert = random_dir * self.epsilon * 0.1
        
        adv_image = adv_image + random_pert
        adv_image = torch.clamp(adv_image, 0, 1)
        
        return adv_image
    
    def _check_adversarial(self, image, original, label, target_label=None):
        """
        Check if image is adversarial
        """
        with torch.no_grad():
            output = self.get_logits(image)
            pred = output.argmax(1)
            orig_output = self.get_logits(original)
            orig_pred = orig_output.argmax(1)
        
        if self.targeted:
            return (pred == target_label).item()
        else:
            return (pred != orig_pred).item()