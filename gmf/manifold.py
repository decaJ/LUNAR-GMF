# Gated Manifold Flow - Manifold Extraction Module
# Implements the extraction of forget and attractor manifolds from activation space

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ForgetManifold:
    """
    Forget Knowledge Manifold: M_f = {a ∈ R^d : a = g_f(z), z ∈ Z_f ⊂ R^k}
    
    Represents the activation patterns associated with knowledge to be forgotten.
    We model this as a Gaussian distribution in activation space for tractability.
    """
    mu: torch.Tensor  # Mean of forget activations: (d_model,)
    Sigma: torch.Tensor  # Covariance matrix: (d_model, d_model) or diagonal (d_model,)
    epsilon: float = 1e-6  # Smoothing term for numerical stability
    
    def distance(self, x: torch.Tensor, method: str = 'mahalanobis') -> torch.Tensor:
        """
        Compute distance from point x to the manifold center.
        
        Args:
            x: Activation tensor of shape (batch_size, d_model) or (d_model,)
            method: 'euclidean' or 'mahalanobis'
        
        Returns:
            Distance tensor of shape (batch_size,) or scalar
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        diff = x - self.mu.unsqueeze(0)  # (batch, d_model)
        
        if method == 'euclidean':
            dist = torch.norm(diff, dim=-1)
        elif method == 'mahalanobis':
            # Regularized inverse covariance for numerical stability
            # Using diagonal approximation for efficiency
            if self.Sigma.dim() == 1:
                # Diagonal covariance
                Sigma_reg = self.Sigma + self.epsilon
                inv_Sigma = 1.0 / Sigma_reg
                mahal_dist = torch.sqrt(torch.sum(diff ** 2 * inv_Sigma.unsqueeze(0), dim=-1))
                dist = mahal_dist
            else:
                # Full covariance - use Woodbury identity for efficiency
                Sigma_reg = self.Sigma + self.epsilon * torch.eye(self.Sigma.size(0), device=self.Sigma.device)
                try:
                    L = torch.linalg.cholesky(Sigma_reg)
                    # Solve L @ y = diff.T, then compute ||y||^2
                    y = torch.linalg.solve_triangular(L, diff.T, upper=False)
                    dist = torch.sqrt(torch.sum(y ** 2, dim=0))
                except RuntimeError:
                    # Fallback to diagonal approximation if Cholesky fails
                    print("Warning: Cholesky decomposition failed, using diagonal approximation")
                    diag_Sigma = torch.diag(self.Sigma) + self.epsilon
                    inv_diag = 1.0 / diag_Sigma
                    dist = torch.sqrt(torch.sum(diff ** 2 * inv_diag.unsqueeze(0), dim=-1))
        else:
            raise ValueError(f"Unknown distance method: {method}")
            
        if squeeze_output:
            return dist.squeeze(0)
        return dist
    
    def to(self, device):
        """Move manifold to Device"""
        self.mu = self.mu.to(device).float()
        self.Sigma = self.Sigma.to(device).float()
        return self


@dataclass
class AttractorManifold:
    """
    Attractor Manifold: M_a = {a ∈ R^d : a = g_a(z), z ∈ Z_a ⊂ R^k}
    
    Represents the target activation patterns for refusal/safe responses.
    This is the manifold we want to push forget activations towards.
    """
    mu: torch.Tensor  # Mean of attractor (refusal) activations: (d_model,)
    direction: torch.Tensor  # Primary refusal direction: (d_model,)
    scale: float = 1.0  # Scaling factor for the attractor
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project activation x towards the attractor manifold.
        
        Args:
            x: Activation tensor of shape (batch_size, d_model)
        
        Returns:
            Projected activation tensor
        """
        # Simple projection: move towards attractor mean along refusal direction
        return x + self.scale * self.direction.unsqueeze(0)
    
    def distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from x to the attractor manifold.
        
        Args:
            x: Activation tensor of shape (batch_size, d_model)
        
        Returns:
            Distance tensor of shape (batch_size,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.norm(x - self.mu.unsqueeze(0), dim=-1)
    
    def to(self, device):
        """Move manifold to device"""
        self.mu = self.mu.to(device).float()
        self.direction = self.direction.to(device).float()
        return self


class ManifoldExtractor:
    """
    Extract forget and attractor manifolds from activation data.
    
    Phase 1: Offline extraction of manifold statistics from forget dataset D_f.
    """
    
    def __init__(
        self,
        hidden_size: int,
        use_diagonal_cov: bool = True,
        epsilon: float = 1e-6,
        device: str = 'cuda'
    ):
        self.hidden_size = hidden_size
        self.use_diagonal_cov = use_diagonal_cov
        self.epsilon = epsilon
        self.device = device
        
    def extract_forget_manifold(
        self,
        activations: List[torch.Tensor],
        normalize: bool = True
    ) -> ForgetManifold:
        """
        Extract the forget manifold from a list of activations.
        
        Args:
            activations: List of activation tensors, each of shape (seq_len, d_model)
                         or (batch, seq_len, d_model)
            normalize: Whether to normalize the covariance
        
        Returns:
            ForgetManifold with computed mu and Sigma
        """
        print("Extracting forget manifold...")
        
        # Concatenate all activations
        all_acts = []
        for act in activations:
            if act.dim() == 3:
                # Take the last token position: (batch, seq_len, d_model) -> (batch, d_model)
                all_acts.append(act[:, -1, :].cpu())
            elif act.dim() == 2:
                # Take the last token: (seq_len, d_model) -> (1, d_model)
                all_acts.append(act[-1:, :].cpu())
            elif act.dim() == 1:
                # Already a vector: (d_model,) -> (1, d_model)
                all_acts.append(act.unsqueeze(0).cpu())
        
        if len(all_acts) == 0:
            raise ValueError("No valid activations provided for manifold extraction")
        
        # Stack and reshape
        concat_acts = torch.cat(all_acts, dim=0)  # (n_samples, d_model)
        print(f"Concatenated activations shape: {concat_acts.shape}")
        
        # Compute mean - ensure float32
        mu = concat_acts.mean(dim=0).float()  # (d_model,)
        
        # Compute covariance - ensure float32
        if self.use_diagonal_cov:
            # Diagonal covariance (variance only) - more efficient
            Sigma = concat_acts.var(dim=0, unbiased=False).float() + self.epsilon
        else:
            # Full covariance matrix
            centered = concat_acts - mu.unsqueeze(0)
            Sigma = (centered.T @ centered) / centered.size(0)
            # Add regularization
            Sigma = Sigma + self.epsilon * torch.eye(self.hidden_size)
        
        print(f"Forget manifold mu shape: {mu.shape}")
        print(f"Forget manifold Sigma shape: {Sigma.shape}")
        
        return ForgetManifold(mu=mu, Sigma=Sigma, epsilon=self.epsilon)
    
    def extract_attractor_manifold(
        self,
        refusal_activations: List[torch.Tensor],
        refusal_direction: torch.Tensor,
        scale: float = 1.0
    ) -> AttractorManifold:
        """
        Extract the attractor manifold from refusal activations.
        
        Args:
            refusal_activations: List of activation tensors from refusal responses
            refusal_direction: Pre-computed refusal direction (from LUNAR-style contrastive)
            scale: Scaling factor for the attractor
        
        Returns:
            AttractorManifold with computed mu and direction
        """
        print("Extracting attractor manifold...")
        
        if len(refusal_activations) > 0:
            # Concatenate all refusal activations
            all_acts = []
            for act in refusal_activations:
                if act.dim() == 3:
                    all_acts.append(act[:, -1, :].cpu())
                elif act.dim() == 2:
                    all_acts.append(act[-1:, :].cpu())
            
            concat_acts = torch.cat(all_acts, dim=0)
            mu = concat_acts.mean(dim=0)
        else:
            # Use the refusal direction as the attractor center
            mu = refusal_direction.clone()
        
        # Normalize the refusal direction
        direction = refusal_direction / (torch.norm(refusal_direction) + self.epsilon)
        
        print(f"Attractor manifold mu shape: {mu.shape}")
        print(f"Attractor direction shape: {direction.shape}")
        
        return AttractorManifold(mu=mu, direction=direction, scale=scale)
    
    @staticmethod
    def compute_refusal_direction(
        harmful_activations: List[torch.Tensor],
        harmless_activations: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the refusal direction using contrastive activations.
        This follows the LUNAR approach for generating candidate directions.
        
        Args:
            harmful_activations: Activations from harmful/forget prompts
            harmless_activations: Activations from harmless/retain prompts
        
        Returns:
            Normalized refusal direction vector
        """
        print("Computing refusal direction...")
        
        def extract_mean(acts):
            all_acts = []
            for act in acts:
                if act.dim() == 3:
                    all_acts.append(act[:, -1, :].cpu())
                elif act.dim() == 2:
                    all_acts.append(act[-1:, :].cpu())
            return torch.cat(all_acts, dim=0).mean(dim=0)
        
        harmful_mean = extract_mean(harmful_activations)
        harmless_mean = extract_mean(harmless_activations)
        
        # Direction from harmless to harmful (refusal direction)
        direction = harmful_mean - harmless_mean
        direction = direction / (torch.norm(direction) + 1e-8)
        
        return direction


class ManifoldDataset:
    """
    Dataset class for storing manifold-related activation data.
    """
    
    def __init__(
        self,
        forget_inputs: List[torch.Tensor],
        forget_targets: Optional[List[torch.Tensor]] = None,
        retain_inputs: Optional[List[torch.Tensor]] = None,
        retain_targets: Optional[List[torch.Tensor]] = None,
    ):
        self.forget_inputs = forget_inputs
        self.forget_targets = forget_targets
        self.retain_inputs = retain_inputs
        self.retain_targets = retain_targets
        
    def __len__(self):
        return len(self.forget_inputs)
    
    def get_forget_batch(self, batch_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get a random batch of forget data"""
        indices = torch.randint(0, len(self.forget_inputs), (batch_size,))
        inputs = torch.stack([self.forget_inputs[i] for i in indices])
        targets = None
        if self.forget_targets is not None:
            targets = torch.stack([self.forget_targets[i] for i in indices])
        return inputs, targets
    
    def get_retain_batch(self, batch_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get a random batch of retain data"""
        if self.retain_inputs is None:
            return None, None
        indices = torch.randint(0, len(self.retain_inputs), (batch_size,))
        inputs = torch.stack([self.retain_inputs[i] for i in indices])
        targets = None
        if self.retain_targets is not None:
            targets = torch.stack([self.retain_targets[i] for i in indices])
        return inputs, targets