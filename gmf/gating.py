# Gated Manifold Flow - Gating Mechanism Module
# Implements distance-based gating for localized manifold transformation

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from .manifold import ForgetManifold


class DistanceBasedGate(nn.Module):
    """
    Distance-based Gating Function (预计算场)
    
    α(x) = exp(-d(x, μ_f)² / 2σ²)
    
    The gate outputs values close to 1 when x is near the forget manifold,
    and close to 0 when x is far from it. This enables:
    - α → 1: Apply full flow transformation (push towards attractor)
    - α → 0: Keep identity mapping (perfect protection of retain knowledge)
    """
    
    def __init__(
        self,
        hidden_size: int,
        sigma: float = 1.0,
        learnable_sigma: bool = False,
        distance_method: str = 'mahalanobis',
        temperature: float = 1.0,
    ):
        """
        Args:
            hidden_size: Dimension of the activation space
            sigma: Initial bandwidth parameter for Gaussian gating
            learnable_sigma: Whether to learn sigma during training
            distance_method: 'euclidean' or 'mahalanobis'
            temperature: Temperature for softening/sharpening the gate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.distance_method = distance_method
        self.temperature = temperature
        
        if learnable_sigma:
            self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        else:
            self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
        
        # Manifold statistics will be set later
        self.register_buffer('manifold_mu', None)
        self.register_buffer('manifold_sigma', None)  # Diagonal covariance
        
    def set_manifold(self, forget_manifold: ForgetManifold):
        """Set the forget manifold statistics (frozen, not learnable)"""
        self.manifold_mu = forget_manifold.mu.clone().detach().float()
        self.manifold_sigma = forget_manifold.Sigma.clone().detach().float()
        print(f"Gate manifold set: mu shape = {self.manifold_mu.shape}")
        
    def compute_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from x to the forget manifold center.
        
        Args:
            x: Activation tensor of shape (batch_size, d_model)
        
        Returns:
            Distance tensor of shape (batch_size,)
        """
        if self.manifold_mu is None:
            raise ValueError("Manifold not set. Call set_manifold() first.")
        
        diff = x - self.manifold_mu.unsqueeze(0)  # (batch, d_model)
        
        if self.distance_method == 'euclidean':
            dist = torch.norm(diff, dim=-1)
        elif self.distance_method == 'mahalanobis':
            if self.manifold_sigma is None:
                # Fallback to Euclidean
                dist = torch.norm(diff, dim=-1)
            else:
                # Diagonal Mahalanobis distance
                # d = sqrt((x-μ)^T Σ^{-1} (x-μ))
                Sigma_reg = self.manifold_sigma + 1e-6
                inv_Sigma = 1.0 / Sigma_reg
                mahal_sq = torch.sum(diff ** 2 * inv_Sigma.unsqueeze(0), dim=-1)
                dist = torch.sqrt(torch.clamp(mahal_sq, min=0))
        else:
            raise ValueError(f"Unknown distance method: {self.distance_method}")
        
        return dist
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gating values for input activations.
        
        Dimension-Normalized Gaussian Gating:
        α(x) = exp(-d(x, μ_f)² / (D · 2σ²))
        
        where D is the feature dimension (hidden_size). This normalization
        makes sigma scale-invariant across different model dimensions.
        
        Physical meaning: We compute "average per-dimension Mahalanobis distance",
        ensuring sigma stays around 1.0 regardless of model dimension.
        
        Args:
            x: Activation tensor of shape (batch_size, d_model)
        
        Returns:
            Gate values α(x) of shape (batch_size, 1) for broadcasting
        """
        dist = self.compute_distance(x)  # (batch,)
        
        # Dimension-normalized Gaussian gating: α(x) = exp(-d² / (D · 2σ²))
        # This prevents underflow in high-dimensional spaces (D=4096 for LLaMA)
        D = self.hidden_size  # Feature dimension (4096 for LLaMA-7B)
        sigma_sq = self.sigma ** 2
        gate = torch.exp(-dist ** 2 / (D * 2 * sigma_sq * self.temperature))
        
        # Return with extra dimension for broadcasting
        return gate.unsqueeze(-1)  # (batch, 1)


class AdaptiveGate(nn.Module):
    """
    Adaptive gating with learned projection.
    
    Instead of pure distance-based gating, this learns a projection
    that can better separate forget and retain regions.
    """
    
    def __init__(
        self,
        hidden_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Learnable projection layers
        layers = []
        in_dim = hidden_size
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.projection = nn.Sequential(*layers)
        self.sigma = sigma
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive gating values.
        
        Args:
            x: Activation tensor of shape (batch_size, d_model)
        
        Returns:
            Gate values of shape (batch_size, 1)
        """
        return self.projection(x)


class GatedFlow(nn.Module):
    """
    Gated Flow Transformation: Residual Gated Mixing
    
    x_new = x + α(x) · (Flow_θ(x) - x)
    
    Physical meaning:
    - When activation is near forget manifold (α → 1): 
        Apply full flow transformation (push towards attractor)
    - When activation is far from forget manifold (α → 0): 
        Keep identity mapping (perfect protection of retain knowledge)
    """
    
    def __init__(
        self,
        flow_transform: nn.Module,
        gate: Union[DistanceBasedGate, AdaptiveGate],
        use_residual: bool = True,
        gate_threshold: float = 1e-8,  # Very small threshold to avoid clamping gate values
    ):
        """
        Args:
            flow_transform: The Flow_θ transformation module
            gate: The gating module (DistanceBasedGate or AdaptiveGate)
            use_residual: Whether to use residual connection
            gate_threshold: Minimum gate value to ensure numerical stability
        """
        super().__init__()
        self.flow_transform = flow_transform
        self.gate = gate
        self.use_residual = use_residual
        self.gate_threshold = gate_threshold
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Apply gated flow transformation.
        
        Args:
            x: Input activation tensor of shape (batch_size, d_model)
        
        Returns:
            Tuple of:
                - Transformed activation tensor
                - Dictionary with gate values and flow outputs for logging
        """
        # Compute gate values
        alpha = self.gate(x)  # (batch, 1)
        
        # Apply threshold for numerical stability
        alpha = torch.clamp(alpha, min=self.gate_threshold)
        
        # Compute flow transformation
        flow_output = self.flow_transform(x)  # (batch, d_model)
        
        # Gated residual mixing: x_new = x + α(x) · (Flow(x) - x)
        if self.use_residual:
            transformed = x + alpha * (flow_output - x)
        else:
            transformed = alpha * flow_output + (1 - alpha) * x
        
        # Return with logging info
        info = {
            'gate_values': alpha.detach(),
            'gate_mean': alpha.mean().item(),
            'gate_std': alpha.std().item(),
            'gate_min': alpha.min().item(),
            'gate_max': alpha.max().item(),
        }
        
        return transformed, info


class MultiLayerGatedFlow(nn.Module):
    """
    Multi-layer Gated Flow for handling multiple transformer layers.
    
    Each layer has its own gated flow transformation, allowing
    layer-specific unlearning behavior.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        flow_hidden_dim: int = 512,
        sigma: float = 1.0,
        learnable_sigma: bool = False,
        distance_method: str = 'mahalanobis',
        share_gates: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create flow transforms for each layer
        self.flow_transforms = nn.ModuleList([
            ResidualFlow(hidden_size, flow_hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Create gates (optionally shared)
        if share_gates:
            shared_gate = DistanceBasedGate(
                hidden_size, sigma, learnable_sigma, distance_method
            )
            self.gates = nn.ModuleList([shared_gate] * num_layers)
        else:
            self.gates = nn.ModuleList([
                DistanceBasedGate(
                    hidden_size, sigma, learnable_sigma, distance_method
                )
                for _ in range(num_layers)
            ])
        
        # Create gated flows
        self.gated_flows = nn.ModuleList([
            GatedFlow(self.flow_transforms[i], self.gates[i])
            for i in range(num_layers)
        ])
        
    def set_manifold(self, forget_manifold: ForgetManifold, layer_idx: Optional[int] = None):
        """
        Set the forget manifold for specific layer or all layers.
        
        Args:
            forget_manifold: The forget manifold to use
            layer_idx: If None, set for all layers; otherwise set for specific layer
        """
        if layer_idx is None:
            for gate in self.gates:
                gate.set_manifold(forget_manifold)
        else:
            self.gates[layer_idx].set_manifold(forget_manifold)
            
    def forward_layer(
        self, 
        x: torch.Tensor, 
        layer_idx: int
    ) -> Tuple[torch.Tensor, dict]:
        """
        Apply gated flow for a specific layer.
        
        Args:
            x: Input activation tensor
            layer_idx: Layer index
        
        Returns:
            Transformed activation and info dict
        """
        return self.gated_flows[layer_idx](x)
    
    def forward(self, x: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, dict]:
        """Alias for forward_layer"""
        return self.forward_layer(x, layer_idx)


# Import ResidualFlow here to avoid circular imports
from .flow_transform import ResidualFlow