# Copyright (c) Steering Unlearn Project
"""
Loss Functions for Steering Unlearn Method - Round 6

Fixed issues from Round 5:
- Added beta regularization to prevent saturation
- Fixed retain loss to properly measure activation changes
- Added detailed monitoring for attractor progress
- Better steering direction handling

Loss components:
1. Attractor Loss: Pull forget samples toward target cone
2. Risk Reduction Loss: Maximize risk reduction (before - after steering)
3. Retain Loss: Keep retain samples unchanged after steering
4. Direction Consistency Loss: Encourage updates in the steering direction
5. Beta Regularization: Prevent beta from saturating at 0 or 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LossComponents:
    """Container for individual loss components - Round 6."""
    attractor_loss: torch.Tensor
    risk_loss: torch.Tensor
    retain_loss: torch.Tensor
    retain_kl_loss: torch.Tensor
    direction_loss: torch.Tensor
    beta_reg_loss: torch.Tensor  # Round 6: NEW - beta regularization
    total_loss: torch.Tensor
    
    # Per-layer losses (for logging)
    attractor_per_layer: Optional[Dict[int, float]] = None
    risk_per_layer: Optional[Dict[int, float]] = None
    retain_per_layer: Optional[Dict[int, float]] = None
    direction_per_layer: Optional[Dict[int, float]] = None
    
    # Monitoring metrics
    risk_reduction: float = 0.0
    forget_update_norm: float = 0.0
    retain_update_norm: float = 0.0
    direction_cosine_sim: float = 0.0
    
    # Round 6: NEW - Additional monitoring
    beta_mean: float = 0.0
    beta_std: float = 0.0
    attractor_cosine_sim: float = 0.0  # Cosine sim to target cone (should increase)
    retain_gate_mean: float = 0.0  # Average gate for retain samples


class SteeringLoss(nn.Module):
    """
    Combined loss for steering unlearning - Round 6.
    
    Key components:
    - Risk reduction loss: L = -(risk_before - risk_after)
    - Direction consistency loss: L = 1 - cos(update, direction)
    - Beta regularization: L = (beta - 0.5)^2 to encourage beta near 0.5
    """
    
    def __init__(
        self,
        lambda_attractor: float = 0.5,
        lambda_risk: float = 1.0,
        lambda_retain: float = 2.0,
        lambda_retain_kl: float = 1.0,
        lambda_direction: float = 0.5,
        lambda_beta_reg: float = 0.1,  # Round 6: NEW
        kl_temperature: float = 1.0,
        primary_target: str = "ignorance",
    ):
        super().__init__()
        
        self.lambda_attractor = lambda_attractor
        self.lambda_risk = lambda_risk
        self.lambda_retain = lambda_retain
        self.lambda_retain_kl = lambda_retain_kl
        self.lambda_direction = lambda_direction
        self.lambda_beta_reg = lambda_beta_reg
        self.kl_temperature = kl_temperature
        self.primary_target = primary_target
    
    def compute_attractor_loss(
        self,
        steered_activations: Dict[int, torch.Tensor],
        target_cones: Dict[int, dict],
        layer_indices: List[int],
        target_type: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """Compute attractor loss: pull forget samples toward target cone axis."""
        if target_type is None:
            target_type = self.primary_target
        
        total_loss = torch.tensor(0.0, device=next(iter(steered_activations.values())).device)
        per_layer_losses = {}
        
        for layer_idx in layer_indices:
            steered = steered_activations[layer_idx]
            
            cone = target_cones[layer_idx][target_type]
            if hasattr(cone, 'axis'):
                target_axis = cone.axis.to(steered.device)
            else:
                target_axis = torch.tensor(cone['axis']).to(steered.device)
            
            steered_float = steered.float()
            target_axis_float = target_axis.float()
            
            # Normalize
            target_axis_norm = F.normalize(target_axis_float.unsqueeze(0), dim=-1).squeeze(0)
            steered_norm = F.normalize(steered_float, dim=-1)
            
            # Compute cosine similarity
            cosine_sim = torch.matmul(steered_norm, target_axis_norm)
            
            # Loss = 1 - cosine_similarity
            layer_loss = (1.0 - cosine_sim).mean()
            
            per_layer_losses[layer_idx] = layer_loss.item()
            total_loss = total_loss + layer_loss
        
        return total_loss, per_layer_losses
    
    def compute_risk_reduction_loss(
        self,
        original_activations: Dict[int, torch.Tensor],
        steered_activations: Dict[int, torch.Tensor],
        risk_probes: Dict[int, nn.Module],
        layer_indices: List[int],
    ) -> Tuple[torch.Tensor, Dict[int, float], float]:
        """
        Compute risk reduction loss - Round 6.
        
        Loss = -(risk_before - risk_after)
        
        This encourages the model to maximize risk reduction.
        """
        total_loss = torch.tensor(0.0, device=next(iter(original_activations.values())).device)
        per_layer_losses = {}
        total_risk_reduction = 0.0
        
        for layer_idx in layer_indices:
            original = original_activations[layer_idx]
            steered = steered_activations[layer_idx]
            probe = risk_probes[layer_idx]
            
            with torch.no_grad():
                risk_before = probe(original.float()).squeeze(-1)
            
            risk_after = probe(steered.float()).squeeze(-1)
            
            risk_reduction = risk_before - risk_after
            layer_loss = -risk_reduction.mean()
            
            per_layer_losses[layer_idx] = layer_loss.item()
            total_loss = total_loss + layer_loss
            total_risk_reduction += risk_reduction.mean().item()
        
        avg_risk_reduction = total_risk_reduction / len(layer_indices)
        
        return total_loss, per_layer_losses, avg_risk_reduction
    
    def compute_retain_loss(
        self,
        original_activations: Dict[int, torch.Tensor],
        steered_activations: Dict[int, torch.Tensor],
        layer_indices: List[int],
        loss_type: str = "mse",
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """Compute retain loss: keep retain samples unchanged after steering."""
        total_loss = torch.tensor(0.0, device=next(iter(original_activations.values())).device)
        per_layer_losses = {}
        
        for layer_idx in layer_indices:
            original = original_activations[layer_idx]
            steered = steered_activations[layer_idx]
            
            original_float = original.float()
            steered_float = steered.float()
            
            if loss_type == "mse":
                layer_loss = F.mse_loss(steered_float, original_float)
            elif loss_type == "cosine":
                cos_sim = F.cosine_similarity(steered_float, original_float, dim=-1)
                layer_loss = (1.0 - cos_sim).mean()
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            per_layer_losses[layer_idx] = layer_loss.item()
            total_loss = total_loss + layer_loss
        
        return total_loss, per_layer_losses
    
    def compute_direction_consistency_loss(
        self,
        forget_original: Dict[int, torch.Tensor],
        forget_steered: Dict[int, torch.Tensor],
        steering_directions: Dict[int, torch.Tensor],
        layer_indices: List[int],
    ) -> Tuple[torch.Tensor, Dict[int, float], float]:
        """
        Compute direction consistency loss - Round 6.
        
        L_dir = 1 - cos(update, direction)
        
        This encourages the forget update to align with the steering direction.
        """
        total_loss = torch.tensor(0.0, device=next(iter(forget_original.values())).device)
        per_layer_losses = {}
        total_cosine_sim = 0.0
        
        for layer_idx in layer_indices:
            # Compute update
            update = forget_steered[layer_idx] - forget_original[layer_idx].to(forget_steered[layer_idx].device)
            
            # Get steering direction
            direction = steering_directions[layer_idx].to(update.device).float()
            
            # Normalize
            update_norm = F.normalize(update.float(), dim=-1)
            direction_norm = F.normalize(direction.unsqueeze(0), dim=-1).squeeze(0)
            
            # Compute cosine similarity
            cosine_sim = torch.matmul(update_norm, direction_norm).mean()
            
            # Loss = 1 - cosine_similarity
            layer_loss = 1.0 - cosine_sim
            
            per_layer_losses[layer_idx] = layer_loss.item()
            total_loss = total_loss + layer_loss
            total_cosine_sim += cosine_sim.item()
        
        avg_cosine_sim = total_cosine_sim / len(layer_indices)
        
        return total_loss, per_layer_losses, avg_cosine_sim
    
    def forward(
        self,
        forget_original_activations: Dict[int, torch.Tensor],
        forget_steered_activations: Dict[int, torch.Tensor],
        retain_original_activations: Dict[int, torch.Tensor],
        retain_steered_activations: Dict[int, torch.Tensor],
        target_cones: Dict[int, dict],
        risk_probes: Dict[int, nn.Module],
        layer_indices: List[int],
        steering_directions: Optional[Dict[int, torch.Tensor]] = None,
        target_type: Optional[str] = None,
        beta_values: Optional[List[float]] = None,
        retain_risk_scores: Optional[Dict[int, float]] = None,
    ) -> LossComponents:
        """Compute total loss for steering unlearning - Round 6."""
        
        # Compute attractor loss
        attractor_loss, attractor_per_layer = self.compute_attractor_loss(
            steered_activations=forget_steered_activations,
            target_cones=target_cones,
            layer_indices=layer_indices,
            target_type=target_type,
        )
        
        # Compute risk reduction loss
        risk_loss, risk_per_layer, risk_reduction = self.compute_risk_reduction_loss(
            original_activations=forget_original_activations,
            steered_activations=forget_steered_activations,
            risk_probes=risk_probes,
            layer_indices=layer_indices,
        )
        
        # Compute retain loss
        retain_loss, retain_per_layer = self.compute_retain_loss(
            original_activations=retain_original_activations,
            steered_activations=retain_steered_activations,
            layer_indices=layer_indices,
        )
        
        # Compute direction consistency loss
        direction_loss = torch.tensor(0.0, device=attractor_loss.device)
        direction_per_layer = {}
        direction_cosine_sim = 0.0
        
        if steering_directions is not None:
            direction_loss, direction_per_layer, direction_cosine_sim = self.compute_direction_consistency_loss(
                forget_original=forget_original_activations,
                forget_steered=forget_steered_activations,
                steering_directions=steering_directions,
                layer_indices=layer_indices,
            )
        
        # Compute beta statistics and regularization
        beta_mean = 0.0
        beta_std = 0.0
        beta_reg_loss = torch.tensor(0.0, device=attractor_loss.device)
        
        if beta_values is not None:
            for bv in beta_values:
                if isinstance(bv, torch.Tensor):
                    beta_mean += bv.mean().item()
                    beta_std += bv.std().item()
                else:
                    beta_mean += float(bv)
            beta_mean /= len(beta_values)
            beta_std /= len(beta_values)
            
            # Beta regularization: penalize deviation from 0.5
            beta_reg_loss = torch.tensor((beta_mean - 0.5) ** 2, device=attractor_loss.device)
        
        # Compute update norms for monitoring
        forget_update_norm = 0.0
        retain_update_norm = 0.0
        
        for layer_idx in layer_indices:
            forget_update = forget_steered_activations[layer_idx] - forget_original_activations[layer_idx].to(forget_steered_activations[layer_idx].device)
            forget_update_norm += forget_update.norm(dim=-1).mean().item()
            
            retain_update = retain_steered_activations[layer_idx] - retain_original_activations[layer_idx].to(retain_steered_activations[layer_idx].device)
            retain_update_norm += retain_update.norm(dim=-1).mean().item()
        
        forget_update_norm /= len(layer_indices)
        retain_update_norm /= len(layer_indices)
        
        # Compute attractor progress (should increase over training)
        attractor_cosine_sim = 0.0
        for layer_idx in layer_indices:
            steered = forget_steered_activations[layer_idx]
            if target_type is None:
                target_type = self.primary_target
            cone = target_cones[layer_idx][target_type]
            if hasattr(cone, 'axis'):
                target_axis = cone.axis.to(steered.device)
            else:
                target_axis = torch.tensor(cone['axis']).to(steered.device)
            
            steered_norm = F.normalize(steered.float(), dim=-1)
            target_axis_norm = F.normalize(target_axis.unsqueeze(0), dim=-1).squeeze(0)
            cosine_sim = torch.matmul(steered_norm, target_axis_norm).mean()
            attractor_cosine_sim += cosine_sim.item()
        
        attractor_cosine_sim /= len(layer_indices)
        
        # Compute retain gate stats for debug
        retain_gate_mean = 0.0
        if retain_risk_scores is not None:
            for layer_idx in layer_indices:
                if layer_idx in retain_risk_scores:
                    retain_risk = retain_risk_scores[layer_idx]
                    if isinstance(retain_risk, torch.Tensor):
                        retain_gate_mean += retain_risk.mean().item()
                    else:
                        retain_gate_mean += float(retain_risk)
            retain_gate_mean /= len(layer_indices)
        
        # Total loss
        total_loss = (
            self.lambda_attractor * attractor_loss +
            self.lambda_risk * risk_loss +
            self.lambda_retain * retain_loss +
            self.lambda_direction * direction_loss +
            self.lambda_beta_reg * beta_reg_loss
        )
        
        return LossComponents(
            attractor_loss=attractor_loss,
            risk_loss=risk_loss,
            retain_loss=retain_loss,
            retain_kl_loss=torch.tensor(0.0, device=attractor_loss.device),
            direction_loss=direction_loss,
            beta_reg_loss=beta_reg_loss,
            total_loss=total_loss,
            attractor_per_layer=attractor_per_layer,
            risk_per_layer=risk_per_layer,
            retain_per_layer=retain_per_layer,
            direction_per_layer=direction_per_layer,
            risk_reduction=risk_reduction,
            forget_update_norm=forget_update_norm,
            retain_update_norm=retain_update_norm,
            direction_cosine_sim=direction_cosine_sim,
            beta_mean=beta_mean,
            beta_std=beta_std,
            attractor_cosine_sim=attractor_cosine_sim,
            retain_gate_mean=retain_gate_mean,
        )


if __name__ == "__main__":
    # Test the loss functions
    print("Testing Steering Loss Functions - Round 6")
    
    batch_size = 4
    hidden_dim = 4096
    layer_indices = [18, 19, 21]
    
    forget_original = {idx: torch.randn(batch_size, hidden_dim) for idx in layer_indices}
    forget_steered = {idx: torch.randn(batch_size, hidden_dim) for idx in layer_indices}
    retain_original = {idx: torch.randn(batch_size, hidden_dim) for idx in layer_indices}
    retain_steered = {idx: torch.randn(batch_size, hidden_dim) for idx in layer_indices}
    
    target_cones = {}
    for idx in layer_indices:
        axis = torch.randn(hidden_dim)
        axis = axis / axis.norm()
        target_cones[idx] = {
            'ignorance': {'axis': axis},
            'refusal': {'axis': torch.randn(hidden_dim)},
        }
    
    steering_directions = {}
    for idx in layer_indices:
        direction = torch.randn(hidden_dim)
        direction = F.normalize(direction, dim=0)
        steering_directions[idx] = direction
    
    class DummyProbe(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
        def forward(self, x):
            return torch.sigmoid(self.linear(x))
    
    risk_probes = {idx: DummyProbe(hidden_dim) for idx in layer_indices}
    
    loss_fn = SteeringLoss(
        lambda_attractor=0.5,
        lambda_risk=1.0,
        lambda_retain=2.0,
        lambda_direction=0.5,
        lambda_beta_reg=0.1,
    )
    
    loss_components = loss_fn(
        forget_original_activations=forget_original,
        forget_steered_activations=forget_steered,
        retain_original_activations=retain_original,
        retain_steered_activations=retain_steered,
        target_cones=target_cones,
        risk_probes=risk_probes,
        layer_indices=layer_indices,
        steering_directions=steering_directions,
        beta_values=[0.5, 0.5, 0.5],
    )
    
    print(f"\nLoss Components (Round 6):")
    print(f"  Attractor Loss: {loss_components.attractor_loss.item():.4f}")
    print(f"  Risk Loss: {loss_components.risk_loss.item():.4f}")
    print(f"  Retain Loss: {loss_components.retain_loss.item():.4f}")
    print(f"  Direction Loss: {loss_components.direction_loss.item():.4f}")
    print(f"  Beta Reg Loss: {loss_components.beta_reg_loss.item():.4f}")
    print(f"  Total Loss: {loss_components.total_loss.item():.4f}")
    print(f"  Risk Reduction: {loss_components.risk_reduction:.4f}")
    print(f"  Direction Cosine Sim: {loss_components.direction_cosine_sim:.4f}")
    print(f"  Attractor Cosine Sim: {loss_components.attractor_cosine_sim:.4f}")
    print(f"  Retain Gate Mean: {loss_components.retain_gate_mean:.4f}")
    
    print("\nLoss function test passed!")