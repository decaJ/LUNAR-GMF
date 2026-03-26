# Copyright (c) Steering Unlearn Project
"""
Steering Module for Steering Unlearn Method - Round 6

Fixed issues from Round 5:
- Beta network now initialized properly with meaningful output range
- Added beta regularization to prevent saturation
- Better steering direction handling

Dual-channel steering:
    update = beta(a) * direction_l + delta_small(a)

where:
    - direction_l: Fixed explicit steering direction
    - beta(a): Scalar strength from a small MLP (initialized to 0.5)
    - delta_small(a): Small residual correction from LoRA
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


class DualChannelSteeringModule(nn.Module):
    """
    Dual-channel steering module - Round 6 (Fixed).
    
    Main update channel: beta(a) * direction_l
        - Provides the main "push" toward target region
        
    Small correction channel: delta_small(a)
        - Provides local adjustments via LoRA
    
    The final update is:
        a_tilde = a + alpha * gate * (beta * direction + delta_small)
    
    Round 6 Fixes:
    - Beta network initialized to output ~0.5 by default (not saturated at 1.0)
    - Better initialization for LoRA layers
    - Track beta statistics for monitoring
    """
    
    def __init__(
        self,
        input_dim: int,
        steering_direction: torch.Tensor,
        rank: int = 64,
        alpha: float = 0.3,
        risk_threshold: float = 0.10,
        delta_norm_clip: float = 1.0,
        beta_hidden_dim: int = 128,
        beta_init_value: float = 0.5,  # Round 6: NEW - initial beta value
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.rank = rank
        self.risk_threshold = risk_threshold
        self.delta_norm_clip = delta_norm_clip
        
        # Register fixed steering direction
        self.register_buffer('steering_direction', steering_direction.clone().float())
        self.register_buffer('alpha', torch.tensor(alpha))
        
        # Beta network: outputs scalar strength for main direction
        # Round 6 FIX: Initialize to output ~beta_init_value instead of saturated at 1.0
        self.beta_hidden = nn.Linear(input_dim, beta_hidden_dim)
        self.beta_output = nn.Linear(beta_hidden_dim, 1)
        
        # Initialize beta network to output ~beta_init_value on average
        # For sigmoid(beta_init_value) = beta_init_value, we need pre-sigmoid = logit(beta_init_value)
        target_logit = torch.log(torch.tensor(beta_init_value) / (1 - beta_init_value + 1e-8) + 1e-8)
        nn.init.zeros_(self.beta_output.weight)
        nn.init.constant_(self.beta_output.bias, target_logit)  # This gives sigmoid(target_logit) ≈ beta_init_value
        nn.init.xavier_uniform_(self.beta_hidden.weight)
        nn.init.zeros_(self.beta_hidden.bias)
        
        # Small residual correction (LoRA)
        self.lora_A = nn.Parameter(torch.zeros(rank, input_dim))
        self.lora_B = nn.Parameter(torch.zeros(input_dim, rank))
        
        # Initialize LoRA with small values (for correction only)
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)  # Round 6: Start with zero LoRA effect
        
        # Layer norm for stable beta computation
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Track beta statistics
        self.beta_std = 0.0  # Track beta variance for monitoring
    
    def compute_gate(self, risk_score: torch.Tensor) -> torch.Tensor:
        """
        Compute gating factor based on risk score and threshold.
        
        Args:
            risk_score: Risk score from probe [batch]
            
        Returns:
            Gate factor [batch, 1] in range [0, 1]
        """
        gate = torch.clamp(
            (risk_score - self.risk_threshold) / (1.0 - self.risk_threshold),
            min=0.0,
            max=1.0
        )
        return gate.unsqueeze(-1)
    
    def compute_beta(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute steering strength beta for the main direction.
        
        Args:
            x: Input activation [batch, hidden_dim] or [batch, seq_len, hidden_dim]
            
        Returns:
            Beta value [batch, 1] or [batch, seq_len, 1]
        """
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)
        
        # Apply layer norm for stability
        x_normed = self.layer_norm(x.float())
        
        # Compute beta - Round 6: Use explicit layers
        hidden = torch.relu(self.beta_hidden(x_normed))
        beta = torch.sigmoid(self.beta_output(hidden))
        
        # Track statistics for monitoring
        with torch.no_grad():
            self.beta_std = beta.std().item()
        
        if len(original_shape) == 3:
            beta = beta.view(batch_size, seq_len, 1)
        
        return beta
    
    def get_beta_stats(self) -> Tuple[float, float]:
        """Get beta statistics (mean and std) for monitoring."""
        with torch.no_grad():
            # Sample a small batch to compute stats
            dummy_input = torch.randn(1, self.input_dim, device=self.steering_direction.device)
            beta = self.compute_beta(dummy_input)
            return beta.mean().item(), self.beta_std
    
    def compute_delta_small(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute small residual correction via LoRA.
        
        Args:
            x: Input activation
            
        Returns:
            Small delta correction
        """
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)
        
        # LoRA forward
        x_float = x.float()
        delta = (x_float @ self.lora_A.T) @ self.lora_B.T
        
        # Optional: clip delta norm
        if self.delta_norm_clip > 0:
            delta_norm = delta.norm(dim=-1, keepdim=True)
            norm_factor = torch.clamp(delta_norm / self.delta_norm_clip, min=1.0)
            delta = delta / (norm_factor + 1e-6) * self.delta_norm_clip
        
        if len(original_shape) == 3:
            delta = delta.view(original_shape)
        
        return delta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute full steering update.
        
        update = beta(x) * direction + delta_small(x)
        
        Args:
            x: Input activation
            
        Returns:
            Steering update (not yet scaled by alpha and gate)
        """
        # Compute beta for main direction
        beta = self.compute_beta(x)
        
        # Main update: beta * direction
        main_update = beta * self.steering_direction
        
        # Small correction: delta_small
        delta_small = self.compute_delta_small(x)
        
        # Combined update
        update = main_update + delta_small
        
        return update
    
    def get_steered_activation(
        self,
        x: torch.Tensor,
        risk_score: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply steering to activation with risk threshold gating.
        
        Args:
            x: Original activation
            risk_score: Risk score from probe [batch]
            alpha: Steering strength (uses self.alpha if None)
            
        Returns:
            Steered activation
        """
        if alpha is None:
            alpha = self.alpha
        
        # Compute gate based on risk
        gate = self.compute_gate(risk_score)
        if x.dim() == 3:
            gate = gate.unsqueeze(-1)  # [batch, 1, 1]
        
        # Compute update
        update = self.forward(x)
        
        # Apply gated steering
        steered = x + alpha * gate * update
        
        return steered


class LoRASteeringModule(nn.Module):
    """
    Legacy LoRA-only steering module (kept for backward compatibility).
    """
    
    def __init__(
        self,
        input_dim: int,
        rank: int = 64,
        scale: float = 1.0,
        alpha: float = 0.08,
        risk_threshold: float = 0.20,
        delta_norm_clip: float = 0.50,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.rank = rank
        self.scale = scale
        self.risk_threshold = risk_threshold
        self.delta_norm_clip = delta_norm_clip
        
        self.lora_A = nn.Parameter(torch.zeros(rank, input_dim))
        self.lora_B = nn.Parameter(torch.zeros(input_dim, rank))
        
        nn.init.normal_(self.lora_A, std=0.1)
        nn.init.normal_(self.lora_B, std=0.1)
        
        self.register_buffer('alpha', torch.tensor(alpha))
    
    def forward(self, x: torch.Tensor, clip_norm: bool = True) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)
        
        input_dtype = x.dtype
        x_for_compute = x.float()
        
        delta = (x_for_compute @ self.lora_A.T) @ self.lora_B.T
        delta = delta * self.scale
        
        if clip_norm and self.delta_norm_clip > 0:
            delta_norm = delta.norm(dim=-1, keepdim=True)
            norm_factor = torch.clamp(delta_norm / self.delta_norm_clip, min=1.0)
            delta = delta / (norm_factor + 1e-6) * self.delta_norm_clip
        
        delta = delta.to(input_dtype)
        
        if len(original_shape) == 3:
            delta = delta.view(original_shape)
        
        return delta
    
    def compute_gate(self, risk_score: torch.Tensor) -> torch.Tensor:
        gate = torch.clamp(
            (risk_score - self.risk_threshold) / (1.0 - self.risk_threshold),
            min=0.0,
            max=1.0
        )
        return gate.unsqueeze(-1)
    
    def get_steered_activation(
        self,
        x: torch.Tensor,
        risk_score: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        if alpha is None:
            alpha = self.alpha
        
        delta = self.forward(x, clip_norm=True)
        gate = self.compute_gate(risk_score)
        
        if x.dim() == 3:
            gate = gate.unsqueeze(-1)
        
        steered = x + alpha * gate * delta
        
        return steered


def create_steering_modules(
    layer_indices: List[int],
    hidden_dim: int,
    steering_directions: Optional[Dict[int, torch.Tensor]] = None,
    module_type: str = "dual_channel",
    rank: int = 64,
    device: str = "cuda",
    alpha_per_layer: Optional[Dict[int, float]] = None,
    risk_threshold: float = 0.10,
    delta_norm_clip: float = 1.0,
    beta_hidden_dim: int = 128,
) -> Dict[int, nn.Module]:
    """
    Create steering modules for all target layers.
    
    Round 5: Creates dual-channel modules by default.
    
    Args:
        layer_indices: List of layer indices
        hidden_dim: Hidden dimension of the model
        steering_directions: Dict mapping layer_idx to fixed steering direction
        module_type: Type of steering module ("dual_channel" or "lora")
        rank: Rank for LoRA modules
        device: Device to place modules on
        alpha_per_layer: Dict mapping layer_idx to alpha value
        risk_threshold: Risk threshold for gating steering
        delta_norm_clip: Maximum delta norm
        beta_hidden_dim: Hidden dimension for beta network
        
    Returns:
        Dictionary mapping layer index to steering module
    """
    modules = {}
    
    # Default alpha values
    default_alphas = {18: 0.1, 19: 0.5, 21: 0.1}  # Round 5: Higher alphas
    
    for layer_idx in layer_indices:
        # Get layer-specific alpha
        if alpha_per_layer is not None and layer_idx in alpha_per_layer:
            alpha = alpha_per_layer[layer_idx]
        elif layer_idx in default_alphas:
            alpha = default_alphas[layer_idx]
        else:
            alpha = 0.3
        
        if module_type == "dual_channel":
            # Get steering direction for this layer
            if steering_directions is None or layer_idx not in steering_directions:
                # Create a random direction as fallback
                direction = torch.randn(hidden_dim)
                direction = F.normalize(direction, dim=0)
                print(f"Warning: No steering direction for layer {layer_idx}, using random direction")
            else:
                direction = steering_directions[layer_idx].to(device)
            
            module = DualChannelSteeringModule(
                input_dim=hidden_dim,
                steering_direction=direction,
                rank=rank,
                alpha=alpha,
                risk_threshold=risk_threshold,
                delta_norm_clip=delta_norm_clip,
                beta_hidden_dim=beta_hidden_dim,
            )
        elif module_type == "lora":
            module = LoRASteeringModule(
                input_dim=hidden_dim,
                rank=rank,
                alpha=alpha,
                risk_threshold=risk_threshold,
                delta_norm_clip=delta_norm_clip,
            )
        else:
            raise ValueError(f"Unknown module type: {module_type}")
        
        modules[layer_idx] = module.to(device)
        print(f"Created {module_type} steering module for layer {layer_idx} with alpha={alpha}")
    
    return modules


def save_steering_modules(
    modules: Dict[int, nn.Module],
    save_dir: str,
):
    """Save all steering modules to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    for layer_idx, module in modules.items():
        module_path = os.path.join(save_dir, f"steering_module_layer{layer_idx}.pt")
        torch.save({
            'layer_idx': layer_idx,
            'state_dict': module.state_dict(),
            'module_type': type(module).__name__,
            'input_dim': module.input_dim,
        }, module_path)
        print(f"Saved steering module for layer {layer_idx}")
    
    combined_path = os.path.join(save_dir, "all_steering_modules.pt")
    combined_data = {
        'layer_indices': list(modules.keys()),
        'modules': {
            f"layer{idx}": {
                'state_dict': module.state_dict(),
                'module_type': type(module).__name__,
                'input_dim': module.input_dim,
            }
            for idx, module in modules.items()
        }
    }
    torch.save(combined_data, combined_path)
    print(f"Saved combined checkpoint to {combined_path}")


def load_steering_modules(
    save_dir: str,
    layer_indices: List[int],
    steering_directions: Optional[Dict[int, torch.Tensor]] = None,
    device: str = "cuda",
) -> Dict[int, nn.Module]:
    """Load steering modules from disk."""
    modules = {}
    
    for layer_idx in layer_indices:
        module_path = os.path.join(save_dir, f"steering_module_layer{layer_idx}.pt")
        checkpoint = torch.load(module_path, map_location=device)
        
        module_type = checkpoint['module_type']
        input_dim = checkpoint['input_dim']
        
        if module_type == "DualChannelSteeringModule":
            if steering_directions is None or layer_idx not in steering_directions:
                direction = torch.randn(input_dim)
                direction = F.normalize(direction, dim=0)
            else:
                direction = steering_directions[layer_idx]
            
            module = DualChannelSteeringModule(
                input_dim=input_dim,
                steering_direction=direction,
            )
        elif module_type == "LoRASteeringModule":
            module = LoRASteeringModule(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown module type: {module_type}")
        
        module.load_state_dict(checkpoint['state_dict'])
        modules[layer_idx] = module.to(device)
        print(f"Loaded steering module for layer {layer_idx}")
    
    return modules


def count_parameters(modules: Dict[int, nn.Module]) -> int:
    """Count total trainable parameters in steering modules."""
    total = 0
    for layer_idx, module in modules.items():
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  Layer {layer_idx}: {n_params:,} parameters")
        total += n_params
    print(f"Total: {total:,} parameters")
    return total


if __name__ == "__main__":
    # Test steering modules
    print("Steering Module Test - Round 5")
    
    batch_size = 4
    hidden_dim = 4096
    layer_indices = [18, 19, 21]
    
    # Create dummy steering directions
    steering_directions = {}
    for idx in layer_indices:
        direction = torch.randn(hidden_dim)
        direction = F.normalize(direction, dim=0)
        steering_directions[idx] = direction
    
    # Create dual-channel modules
    print("\nCreating Dual-Channel steering modules...")
    modules = create_steering_modules(
        layer_indices=layer_indices,
        hidden_dim=hidden_dim,
        steering_directions=steering_directions,
        module_type="dual_channel",
        rank=64,
    )
    
    # Count parameters
    print("\nParameter count:")
    count_parameters(modules)
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(batch_size, hidden_dim)
    risk_score = torch.rand(batch_size)
    
    for layer_idx, module in modules.items():
        update = module(x)
        steered = module.get_steered_activation(x, risk_score)
        print(f"  Layer {layer_idx}: update shape={update.shape}, steered shape={steered.shape}")
    
    print("\nSteering modules working correctly!")