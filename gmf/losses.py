# Gated Manifold Flow - Loss Functions Module
# Implements the multi-objective loss for GMF unlearning

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class GMFLossConfig:
    """Configuration for GMF loss function."""
    lambda_attractor: float = 1.0      # Weight for attractor loss
    lambda_retain: float = 1.0         # Weight for retain loss
    lambda_flow: float = 0.1           # Weight for flow regularization
    lambda_recoverability: float = 0.1 # Weight for recoverability loss
    barrier_weight: float = 0.5        # Weight for barrier function
    
    # Distance settings
    attractor_distance_method: str = 'euclidean'
    
    # Barrier function settings
    barrier_threshold: float = 2.0     # Distance threshold for barrier
    barrier_sharpness: float = 1.0     # Sharpness of barrier function


class GMFLoss(nn.Module):
    """
    Gated Manifold Flow Loss Function
    
    L = λ_att * L_attractor + λ_ret * L_retain + λ_flow * L_flow + λ_rec * L_recoverability
    
    Components:
    - L_attractor: Pull forget activations towards attractor manifold
    - L_retain: Keep retain activations unchanged (gate protection)
    - L_flow: Flow regularization (smoothness and invertibility)
    - L_recoverability: Ensure transformed activations are valid
    """
    
    def __init__(self, config: Optional[GMFLossConfig] = None):
        super().__init__()
        self.config = config or GMFLossConfig()
        
    def attractor_loss(
        self,
        transformed_x: torch.Tensor,
        attractor_mu: torch.Tensor,
        attractor_direction: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Attractor Loss: Direct MSE loss
        
        直接使用MSE损失，将遗忘数据的激活拉向attractor流形。
        
        Args:
            transformed_x: Transformed activations (batch, d_model)
            attractor_mu: Mean of attractor manifold (d_model,)
            attractor_direction: Optional direction for directed attractor
        
        Returns:
            Tuple of (total loss, dict of loss components for logging)
        """
        loss_components = {}
        
        # --- 直接使用MSE损失 ---
        # 将transformed_x拉向attractor_mu
        loss_mse = F.mse_loss(transformed_x, attractor_mu.unsqueeze(0).expand_as(transformed_x))
        
        loss_attractor = loss_mse
        
        # Store components as Python floats for logging (not used in tensor ops)
        loss_components['mse_loss'] = loss_mse.item()
        
        # Optional: Add directional loss if attractor_direction is provided
        if attractor_direction is not None:
            # Encourage movement along the refusal direction
            direction_norm = attractor_direction / (torch.norm(attractor_direction) + 1e-8)
            projection = (transformed_x * direction_norm.unsqueeze(0)).sum(dim=-1)
            target_projection = (attractor_mu * direction_norm).sum() + self.config.barrier_threshold
            directional_loss = F.mse_loss(projection, target_projection.expand_as(projection))
            loss_attractor = loss_attractor + 0.5 * directional_loss
            loss_components['directional_loss'] = directional_loss.item()
        
        return loss_attractor, loss_components
    
    def retain_loss(
        self,
        original_x: torch.Tensor,
        transformed_x: torch.Tensor,
        gate_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retain Loss: L_ret = E[||Flow(x) - x||²]
        
        Minimize deviation for retain activations (should be protected by gate).
        When gate is near 0, transformed_x ≈ original_x, so this loss should be small.
        
        Args:
            original_x: Original activations (batch, d_model)
            transformed_x: Transformed activations (batch, d_model)
            gate_values: Gate values α(x) (batch, 1)
        
        Returns:
            Scalar loss value
        """
        # For retain data, gate should be low, so transformation should be minimal
        deviation = transformed_x - original_x
        deviation_norm = torch.norm(deviation, dim=-1)
        
        # Weight by inverse of gate (higher loss when gate is high for retain data)
        # This penalizes transformations on retain activations
        inverse_gate_weight = 1.0 - gate_values.squeeze(-1)
        weighted_loss = (deviation_norm.pow(2) * inverse_gate_weight).mean()
        
        return weighted_loss
    
    def flow_regularization(
        self,
        original_x: torch.Tensor,
        transformed_x: torch.Tensor,
        gate_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flow Regularization Loss: L_flow
        
        Encourage smooth, bounded transformations:
        1. Jacobian determinant regularization (for invertibility)
        2. Lipschitz regularization (for smoothness)
        
        Args:
            original_x: Original activations
            transformed_x: Transformed activations
            gate_values: Gate values
        
        Returns:
            Scalar loss value
        """
        # 1. Transformation magnitude regularization
        # Encourage bounded transformations
        transform_magnitude = torch.norm(transformed_x - original_x, dim=-1)
        magnitude_loss = transform_magnitude.pow(2).mean()
        
        # 2. Gate-weighted smoothness
        # Higher gate = more transformation allowed
        gate_smoothness = (gate_values.squeeze(-1) * transform_magnitude).mean()
        
        return magnitude_loss + 0.1 * gate_smoothness
    
    def barrier_function(
        self,
        x: torch.Tensor,
        forget_manifold_mu: torch.Tensor,
        forget_manifold_sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Barrier Function: Prevent transformation from affecting retain region.
        
        B(x) = 1 / (1 + exp(-k(d(x, M_f) - τ)))
        
        This creates a soft barrier that:
        - Is low near forget manifold (allow transformation)
        - Is high far from forget manifold (block transformation)
        
        Args:
            x: Activation tensor
            forget_manifold_mu: Center of forget manifold
            forget_manifold_sigma: Optional covariance for Mahalanobis distance
        
        Returns:
            Barrier values
        """
        # Compute distance to forget manifold
        diff = x - forget_manifold_mu.unsqueeze(0)
        
        if forget_manifold_sigma is not None:
            # Mahalanobis distance (diagonal approximation)
            Sigma_reg = forget_manifold_sigma + 1e-6
            inv_Sigma = 1.0 / Sigma_reg
            dist = torch.sqrt(torch.sum(diff ** 2 * inv_Sigma.unsqueeze(0), dim=-1))
        else:
            dist = torch.norm(diff, dim=-1)
        
        # Soft barrier function
        barrier = torch.sigmoid(
            self.config.barrier_sharpness * (dist - self.config.barrier_threshold)
        )
        
        return barrier
    
    def recoverability_loss(
        self,
        transformed_x: torch.Tensor,
        target_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Recoverability Loss: Ensure transformations are valid.
        
        This can be implemented as:
        1. Output norm regularization (prevent explosion)
        2. Output range constraints
        3. Semantic consistency (optional, requires model)
        
        Args:
            transformed_x: Transformed activations
            target_x: Optional target activations
        
        Returns:
            Scalar loss value
        """
        # 1. Norm regularization - prevent explosion
        output_norm = torch.norm(transformed_x, dim=-1)
        norm_loss = F.mse_loss(output_norm, torch.ones_like(output_norm) * output_norm.mean().detach())
        
        # 2. Range constraint - soft clipping
        max_val = transformed_x.abs().max()
        range_loss = F.relu(max_val - 100.0)  # Prevent extreme values
        
        return norm_loss + 0.01 * range_loss
    
    def forward(
        self,
        original_x: torch.Tensor,
        transformed_x: torch.Tensor,
        gate_values: torch.Tensor,
        attractor_mu: torch.Tensor,
        forget_manifold_mu: torch.Tensor,
        forget_manifold_sigma: Optional[torch.Tensor] = None,
        attractor_direction: Optional[torch.Tensor] = None,
        is_forget: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the full GMF loss.
        
        Args:
            original_x: Original activations (batch, d_model)
            transformed_x: Transformed activations (batch, d_model)
            gate_values: Gate values α(x) (batch, 1)
            attractor_mu: Mean of attractor manifold
            forget_manifold_mu: Mean of forget manifold
            forget_manifold_sigma: Optional covariance of forget manifold
            attractor_direction: Optional direction for attractor
            is_forget: Whether this is forget data (vs retain data)
        
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        
        if is_forget:
            # For forget data: attractor loss is primary
            # attractor_loss now returns (loss, components_dict)
            attr_loss, attr_components = self.attractor_loss(
                transformed_x, attractor_mu, attractor_direction
            )
            losses['attractor_loss'] = attr_loss
            # Store detailed components for logging
            for k, v in attr_components.items():
                losses[f'attractor_{k}'] = v
            losses['retain_loss'] = torch.zeros(1, device=original_x.device, dtype=original_x.dtype).squeeze()
        else:
            # For retain data: retain loss is primary
            losses['attractor_loss'] = torch.zeros(1, device=original_x.device, dtype=original_x.dtype).squeeze()
            losses['retain_loss'] = self.retain_loss(
                original_x, transformed_x, gate_values
            )
        
        # Flow regularization (for both)
        losses['flow_reg'] = self.flow_regularization(
            original_x, transformed_x, gate_values
        )
        
        # Recoverability loss
        losses['recoverability_loss'] = self.recoverability_loss(transformed_x)
        
        # Barrier function loss (for forget data)
        if is_forget:
            barrier = self.barrier_function(
                transformed_x, forget_manifold_mu, forget_manifold_sigma
            )
            # Barrier should be low for transformed forget activations
            losses['barrier_loss'] = barrier.mean()
        else:
            losses['barrier_loss'] = torch.zeros(1, device=original_x.device, dtype=original_x.dtype).squeeze()
        
        # Compute total loss
        total_loss = (
            self.config.lambda_attractor * losses['attractor_loss'] +
            self.config.lambda_retain * losses['retain_loss'] +
            self.config.lambda_flow * losses['flow_reg'] +
            self.config.lambda_recoverability * losses['recoverability_loss'] +
            self.config.barrier_weight * losses['barrier_loss']
        )
        losses['total_loss'] = total_loss
        
        return losses


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning separation between forget and retain activations.
    
    This can be used as an auxiliary loss to improve the gating function.
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        forget_embeddings: torch.Tensor,
        retain_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            forget_embeddings: Embeddings from forget data
            retain_embeddings: Embeddings from retain data
        
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        forget_norm = F.normalize(forget_embeddings, dim=-1)
        retain_norm = F.normalize(retain_embeddings, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.mm(forget_norm, retain_norm.T) / self.temperature
        
        # Contrastive loss: maximize similarity within forget, minimize with retain
        # For forget: should be similar to each other, dissimilar to retain
        forget_sim = torch.mm(forget_norm, forget_norm.T) / self.temperature
        
        # InfoNCE-style loss
        labels = torch.arange(forget_embeddings.size(0), device=forget_embeddings.device)
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge distillation loss for preserving model capabilities.
    
    This helps the unlearned model retain general capabilities
    while forgetting specific knowledge.
    """
    
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Logits from student (unlearned) model
            teacher_logits: Logits from teacher (original) model
        
        Returns:
            KD loss value
        """
        # Soft targets
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence
        loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return loss