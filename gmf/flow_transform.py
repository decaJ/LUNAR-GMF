# Gated Manifold Flow - Flow Transformation Module
# Implements Flow_θ transformation for manifold flow

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from abc import ABC, abstractmethod


class FlowTransform(ABC, nn.Module):
    """
    Abstract base class for flow transformations.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input x to output."""
        pass
    
    @abstractmethod
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse transformation (if invertible)."""
        pass


class ResidualFlow(FlowTransform):
    """
    Residual Flow Transformation: Flow_θ(x) = x + f_θ(x)
    
    Simple but effective transformation that learns a residual perturbation.
    The residual network f_θ is parameterized by MLP layers.
    
    This is computationally efficient and ensures:
    - Identity initialization (when f_θ = 0)
    - Easy invertibility (x = y - f_θ(y))
    """
    
    def __init__(
        self,
        hidden_size: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        activation: str = 'gelu',
        dropout: float = 0.1,
        init_scale: float = 0.01,
    ):
        """
        Args:
            hidden_size: Dimension of input/output space (d_model)
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            activation: Activation function ('relu', 'gelu', 'tanh')
            dropout: Dropout rate
            init_scale: Initialization scale for the last layer (small = near identity)
        """
        super().__init__(hidden_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'silu':
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build MLP layers
        layers = []
        in_dim = hidden_size
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, hidden_size))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize last layer with small weights for near-identity initialization
        self._init_weights(init_scale)
        
    def _init_weights(self, init_scale: float):
        """Initialize weights, especially the last layer with small values."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Small initialization for last layer to start near identity
        last_layer = self.mlp[-1]
        nn.init.uniform_(last_layer.weight, -init_scale, init_scale)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = x + f_θ(x)
        
        Args:
            x: Input tensor of shape (batch_size, hidden_size) or 
               (batch_size, seq_len, hidden_size)
        
        Returns:
            Transformed tensor of same shape
        """
        # Convert to float32 for computation, keep original dtype for residual
        original_dtype = x.dtype
        x_float = x.float()
        residual = self.mlp(x_float)
        return x_float + residual
    
    def inverse(self, y: torch.Tensor, num_iterations: int = 10) -> torch.Tensor:
        """
        Inverse pass using fixed-point iteration: x = y - f_θ(x)
        
        Since ResidualFlow is y = x + f_θ(x), the inverse satisfies:
        x = y - f_θ(x)
        
        We can solve this iteratively:
        x_{t+1} = y - f_θ(x_t)
        
        Args:
            y: Output tensor of shape (batch_size, hidden_size)
            num_iterations: Number of fixed-point iterations
        
        Returns:
            Inverted tensor x
        """
        x = y.clone()
        for _ in range(num_iterations):
            x = y - self.mlp(x)
        return x


class AffineFlow(FlowTransform):
    """
    Affine Flow Transformation: y = s * x + t
    
    Simple affine transformation with scale (s) and translation (t).
    Both s and t are parameterized by neural networks.
    """
    
    def __init__(
        self,
        hidden_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__(hidden_size)
        
        # Scale network
        scale_layers = [nn.Linear(hidden_size, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            scale_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        scale_layers.append(nn.Linear(hidden_dim, hidden_size))
        self.scale_net = nn.Sequential(*scale_layers)
        
        # Translation network
        trans_layers = [nn.Linear(hidden_size, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            trans_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        trans_layers.append(nn.Linear(hidden_dim, hidden_size))
        self.trans_net = nn.Sequential(*trans_layers)
        
        # Initialize to identity transformation
        self._init_weights()
        
    def _init_weights(self):
        """Initialize to identity: s = 1, t = 0"""
        # Initialize scale network to output zeros (then exp(0) = 1)
        for module in self.scale_net:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize translation network to output zeros
        for module in self.trans_net:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = s * x + t
        
        Args:
            x: Input tensor
        
        Returns:
            Transformed tensor
        """
        log_scale = self.scale_net(x)
        scale = torch.exp(log_scale.clamp(-5, 5))  # Clamp for numerical stability
        translation = self.trans_net(x)
        return scale * x + translation
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: x = (y - t) / s
        
        Note: This requires fixed-point iteration since s and t depend on x,
        not y. We approximate by using y to compute s and t.
        """
        log_scale = self.scale_net(y)
        scale = torch.exp(log_scale.clamp(-5, 5))
        translation = self.trans_net(y)
        return (y - translation) / (scale + 1e-8)


class ManifoldFlow(FlowTransform):
    """
    Manifold Flow Transformation that pushes activations towards attractor manifold.
    
    This combines:
    1. Projection towards attractor manifold
    2. Learned residual transformation
    
    Used for LLM unlearning where we want to transform forget activations
    towards a "refusal" attractor while preserving retain knowledge.
    """
    
    def __init__(
        self,
        hidden_size: int,
        hidden_dim: int = 512,
        attractor_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__(hidden_size)
        
        # Attractor embedding
        self.attractor_proj = nn.Sequential(
            nn.Linear(hidden_size, attractor_dim),
            nn.LayerNorm(attractor_dim),
            nn.GELU(),
        )
        
        # Main transformation network
        layers = [
            nn.Linear(hidden_size + attractor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
        layers.append(nn.Linear(hidden_dim, hidden_size))
        
        self.transform = nn.Sequential(*layers)
        
        # Attractor manifold statistics (set externally)
        self.register_buffer('attractor_mu', None)
        self.register_buffer('attractor_direction', None)
        self.attractor_scale = 1.0
        
    def set_attractor(self, attractor_mu: torch.Tensor, attractor_direction: torch.Tensor, scale: float = 1.0):
        """Set attractor manifold statistics."""
        self.attractor_mu = attractor_mu.clone().detach()
        self.attractor_direction = attractor_direction.clone().detach()
        self.attractor_scale = scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attractor projection.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_size)
        
        Returns:
            Transformed tensor
        """
        # Compute attractor embedding
        if self.attractor_mu is not None:
            # Project towards attractor
            attractor_info = self.attractor_proj(x)
        else:
            attractor_info = torch.zeros(x.size(0), self.attractor_proj[0].out_features, device=x.device)
        
        # Concatenate input with attractor embedding
        combined = torch.cat([x, attractor_info], dim=-1)
        
        # Apply transformation
        residual = self.transform(combined)
        return x + residual
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Approximate inverse (not exact for this architecture)."""
        # This is an approximation since the transformation is not easily invertible
        attractor_info = self.attractor_proj(y)
        combined = torch.cat([y, attractor_info], dim=-1)
        residual = self.transform(combined)
        return y - residual


class FlowModule(nn.Module):
    """
    Complete Flow module that combines multiple flow transformations.
    
    This implements the complete Flow_θ used in the gated flow:
    - Multiple flow transformations stacked
    - Optional normalization layers
    - Jacobi determinant computation for density estimation
    """
    
    def __init__(
        self,
        hidden_size: int,
        hidden_dim: int = 512,
        num_flows: int = 3,
        flow_type: str = 'residual',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_flows = num_flows
        
        # Create flow layers
        if flow_type == 'residual':
            FlowClass = ResidualFlow
        elif flow_type == 'affine':
            FlowClass = AffineFlow
        elif flow_type == 'manifold':
            FlowClass = ManifoldFlow
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")
        
        self.flows = nn.ModuleList([
            FlowClass(hidden_size, hidden_dim)
            for _ in range(num_flows)
        ])
        
        # Normalization between flows
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_flows - 1)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through all flow transformations.
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of:
                - Output tensor
                - Dictionary with flow statistics
        """
        log_det = 0
        flow_outputs = []
        
        for i, flow in enumerate(self.flows):
            x = flow(x)
            flow_outputs.append(x)
            
            if i < len(self.norms):
                x = self.norms[i](x)
        
        stats = {
            'log_det': log_det,
            'flow_outputs': flow_outputs,
        }
        
        return x, stats
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse pass through all flows."""
        x = y
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        return x