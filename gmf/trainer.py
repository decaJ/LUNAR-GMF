# Gated Manifold Flow - Trainer Module
# Implements the training pipeline for GMF unlearning

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm
import os
import json
import logging

from .manifold import ForgetManifold, AttractorManifold, ManifoldExtractor
from .gating import DistanceBasedGate, GatedFlow
from .flow_transform import ResidualFlow, FlowTransform
from .losses import GMFLoss, GMFLossConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GMFTrainerConfig:
    """Configuration for GMF Trainer."""
    # Training parameters
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Gating parameters
    sigma: float = 1.0
    learnable_sigma: bool = False
    distance_method: str = 'mahalanobis'
    
    # Flow parameters
    flow_hidden_dim: int = 512
    flow_num_layers: int = 3
    flow_type: str = 'residual'
    
    # Loss parameters
    lambda_attractor: float = 1.0
    lambda_retain: float = 1.0
    lambda_flow: float = 0.1
    lambda_recoverability: float = 0.1
    
    # Checkpointing
    save_dir: str = "checkpoints/gmf"
    save_every: int = 5
    log_every: int = 10
    
    # Device
    device: str = 'cuda'


class GMFModule(nn.Module):
    """
    Complete GMF module combining flow transform and gating.
    
    This module is applied to activations at specific layers
    to implement the gated manifold flow transformation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        config: GMFTrainerConfig,
        device: str = 'cuda',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.device = device
        
        # Create flow transformation
        self.flow_transform = ResidualFlow(
            hidden_size=hidden_size,
            hidden_dim=config.flow_hidden_dim,
            num_layers=config.flow_num_layers,
        ).to(device)
        
        # Create gating function
        self.gate = DistanceBasedGate(
            hidden_size=hidden_size,
            sigma=config.sigma,
            learnable_sigma=config.learnable_sigma,
            distance_method=config.distance_method,
        ).to(device)
        
        # Create gated flow
        self.gated_flow = GatedFlow(
            flow_transform=self.flow_transform,
            gate=self.gate,
        ).to(device)
        
    def set_manifold(self, forget_manifold: ForgetManifold):
        """Set the forget manifold for gating."""
        self.gate.set_manifold(forget_manifold)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Apply gated flow transformation."""
        return self.gated_flow(x)


class GMFTrainer:
    """
    Trainer for Gated Manifold Flow unlearning.
    
    Two-phase training:
    1. Phase 1: Extract manifold statistics (offline)
    2. Phase 2: Train the gated flow transformation
    """
    
    def __init__(
        self,
        hidden_size: int,
        config: Optional[GMFTrainerConfig] = None,
    ):
        self.hidden_size = hidden_size
        self.config = config or GMFTrainerConfig()
        self.device = self.config.device
        
        # Create GMF module
        self.gmf_module = GMFModule(hidden_size, self.config, device=self.device).to(self.device)
        
        # Create loss function
        loss_config = GMFLossConfig(
            lambda_attractor=self.config.lambda_attractor,
            lambda_retain=self.config.lambda_retain,
            lambda_flow=self.config.lambda_flow,
            lambda_recoverability=self.config.lambda_recoverability,
        )
        self.loss_fn = GMFLoss(loss_config)
        
        # Manifold statistics (set in Phase 1)
        self.forget_manifold: Optional[ForgetManifold] = None
        self.attractor_manifold: Optional[AttractorManifold] = None
        
        # Optimizer (created in Phase 2)
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'forget_loss': [],
            'retain_loss': [],
            'gate_mean': [],
        }
        
    def phase1_extract_manifolds(
        self,
        forget_activations: List[torch.Tensor],
        retain_activations: List[torch.Tensor],
        refusal_direction: torch.Tensor,
    ) -> Tuple[ForgetManifold, AttractorManifold]:
        """
        Phase 1: Extract forget and attractor manifold statistics.
        
        This is done offline before training the flow transformation.
        
        Args:
            forget_activations: Activations from forget dataset
            retain_activations: Activations from retain dataset
            refusal_direction: Pre-computed refusal direction
        
        Returns:
            Tuple of (ForgetManifold, AttractorManifold)
        """
        logger.info("=== Phase 1: Extracting Manifold Statistics ===")
        
        # Extract forget manifold
        extractor = ManifoldExtractor(
            hidden_size=self.hidden_size,
            use_diagonal_cov=True,
            device=self.device,
        )
        
        self.forget_manifold = extractor.extract_forget_manifold(forget_activations)
        self.forget_manifold = self.forget_manifold.to(self.device)
        
        # Create attractor manifold using refusal direction
        refusal_direction = refusal_direction.to(self.device)
        self.attractor_manifold = AttractorManifold(
            mu=self.forget_manifold.mu + refusal_direction * 2.0,  # Offset along refusal direction
            direction=refusal_direction,
            scale=1.0,
        )
        self.attractor_manifold = self.attractor_manifold.to(self.device)
        
        # Set manifold in GMF module
        self.gmf_module.set_manifold(self.forget_manifold)
        
        logger.info(f"Forget manifold mean norm: {torch.norm(self.forget_manifold.mu):.4f}")
        logger.info(f"Attractor manifold mean norm: {torch.norm(self.attractor_manifold.mu):.4f}")
        
        return self.forget_manifold, self.attractor_manifold
    
    def phase2_train(
        self,
        forget_inputs: torch.Tensor,
        retain_inputs: torch.Tensor,
        forget_targets: Optional[torch.Tensor] = None,
        retain_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Phase 2: Train the gated flow transformation.
        
        Args:
            forget_inputs: Forget dataset inputs (n_forget, d_model)
            retain_inputs: Retain dataset inputs (n_retain, d_model)
            forget_targets: Optional target activations for forget data
            retain_targets: Optional target activations for retain data
        
        Returns:
            Training statistics
        """
        logger.info("=== Phase 2: Training Gated Flow ===")
        
        # Move inputs to device and ensure float32 dtype
        forget_inputs = forget_inputs.to(self.device).float()
        retain_inputs = retain_inputs.to(self.device).float()
        if forget_targets is not None:
            forget_targets = forget_targets.to(self.device).float()
        if retain_targets is not None:
            retain_targets = retain_targets.to(self.device).float()
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.gmf_module.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Create scheduler
        # Calculate total steps (add 1 to avoid off-by-one error)
        steps_per_epoch = max(
            (len(forget_inputs) + self.config.batch_size - 1) // self.config.batch_size,
            (len(retain_inputs) + self.config.batch_size - 1) // self.config.batch_size,
        )
        total_steps = self.config.num_epochs * steps_per_epoch + 1
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
        )
        
        # Create data loaders
        forget_dataset = ActivationDataset(forget_inputs, forget_targets)
        retain_dataset = ActivationDataset(retain_inputs, retain_targets)
        
        forget_loader = DataLoader(
            forget_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        retain_loader = DataLoader(
            retain_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        # Training loop
        self.gmf_module.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            forget_losses = []
            retain_losses = []
            gate_means = []
            
            # Alternate between forget and retain batches
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)
            
            num_batches = max(len(forget_loader), len(retain_loader))
            
            for batch_idx in range(num_batches):
                # Get forget batch
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_batch = next(forget_iter)
                
                # Get retain batch
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batch = next(retain_iter)
                
                # Process forget batch
                self.optimizer.zero_grad()
                
                forget_input = forget_batch['input'].to(self.device)
                forget_transformed, forget_info = self.gmf_module(forget_input)
                
                forget_loss_dict = self.loss_fn(
                    original_x=forget_input,
                    transformed_x=forget_transformed,
                    gate_values=forget_info['gate_values'],
                    attractor_mu=self.attractor_manifold.mu.float(),
                    forget_manifold_mu=self.forget_manifold.mu.float(),
                    forget_manifold_sigma=self.forget_manifold.Sigma.float(),
                    attractor_direction=self.attractor_manifold.direction.float() if self.attractor_manifold.direction is not None else None,
                    is_forget=True,
                )
                forget_loss = forget_loss_dict['total_loss']
                
                # Process retain batch
                retain_input = retain_batch['input'].to(self.device)
                retain_transformed, retain_info = self.gmf_module(retain_input)
                
                retain_loss_dict = self.loss_fn(
                    original_x=retain_input,
                    transformed_x=retain_transformed,
                    gate_values=retain_info['gate_values'],
                    attractor_mu=self.attractor_manifold.mu.float(),
                    forget_manifold_mu=self.forget_manifold.mu.float(),
                    forget_manifold_sigma=self.forget_manifold.Sigma.float(),
                    attractor_direction=self.attractor_manifold.direction.float() if self.attractor_manifold.direction is not None else None,
                    is_forget=False,
                )
                retain_loss = retain_loss_dict['total_loss']
                
                # Combined loss
                total_loss = forget_loss + retain_loss
                
                # Backward and optimize
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.gmf_module.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                
                # Logging
                epoch_losses.append(total_loss.item())
                forget_losses.append(forget_loss.item())
                retain_losses.append(retain_loss.item())
                gate_means.append(forget_info['gate_mean'])
            
            # Epoch statistics
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_forget_loss = sum(forget_losses) / len(forget_losses)
            avg_retain_loss = sum(retain_losses) / len(retain_losses)
            avg_gate = sum(gate_means) / len(gate_means)
            
            self.history['train_loss'].append(avg_loss)
            self.history['forget_loss'].append(avg_forget_loss)
            self.history['retain_loss'].append(avg_retain_loss)
            self.history['gate_mean'].append(avg_gate)
            
            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, "
                       f"forget={avg_forget_loss:.4f}, retain={avg_retain_loss:.4f}, "
                       f"gate={avg_gate:.3f}")
            
            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        logger.info("Training completed!")
        return self.history
    
    def transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply the trained transformation to new activations.
        
        Args:
            x: Input activations (batch, d_model)
        
        Returns:
            Tuple of (transformed activations, info dict)
        """
        self.gmf_module.eval()
        with torch.no_grad():
            x = x.to(self.device)
            transformed, info = self.gmf_module(x)
        return transformed, info
    
    def save_checkpoint(self, epoch: int):
        """Save a training checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.save_dir, 
            f"gmf_checkpoint_epoch{epoch}.pt"
        )
        
        checkpoint = {
            'epoch': epoch,
            'gmf_module_state_dict': self.gmf_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config,
            'forget_manifold_mu': self.forget_manifold.mu if self.forget_manifold else None,
            'forget_manifold_sigma': self.forget_manifold.Sigma if self.forget_manifold else None,
            'attractor_manifold_mu': self.attractor_manifold.mu if self.attractor_manifold else None,
            'attractor_manifold_direction': self.attractor_manifold.direction if self.attractor_manifold else None,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.gmf_module.load_state_dict(checkpoint['gmf_module_state_dict'])
        
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint['history']
        
        # Restore manifold statistics
        if checkpoint['forget_manifold_mu'] is not None:
            self.forget_manifold = ForgetManifold(
                mu=checkpoint['forget_manifold_mu'],
                Sigma=checkpoint['forget_manifold_sigma'],
            )
            self.gmf_module.set_manifold(self.forget_manifold)
        
        if checkpoint['attractor_manifold_mu'] is not None:
            self.attractor_manifold = AttractorManifold(
                mu=checkpoint['attractor_manifold_mu'],
                direction=checkpoint['attractor_manifold_direction'],
            )
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
    
    def get_flow_weights(self) -> Dict[str, torch.Tensor]:
        """Get the trained flow transformation weights for integration with LLM."""
        return {
            'flow_transform': self.gmf_module.flow_transform.state_dict(),
            'gate': self.gmf_module.gate.state_dict(),
        }


class ActivationDataset(Dataset):
    """Simple dataset for activation data."""
    
    def __init__(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        self.inputs = inputs
        self.targets = targets
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        item = {'input': self.inputs[idx]}
        if self.targets is not None:
            item['target'] = self.targets[idx]
        return item


class MultiLayerGMFTrainer:
    """
    Trainer for multi-layer GMF unlearning.
    
    Handles training GMF modules for multiple transformer layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        layer_indices: List[int],
        config: Optional[GMFTrainerConfig] = None,
    ):
        self.hidden_size = hidden_size
        self.layer_indices = layer_indices
        self.config = config or GMFTrainerConfig()
        
        # Create a GMF trainer for each layer
        self.trainers = {
            layer_idx: GMFTrainer(hidden_size, self.config)
            for layer_idx in layer_indices
        }
        
    def phase1_extract_manifolds(
        self,
        layer_activations: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]],
        refusal_directions: Dict[int, torch.Tensor],
    ):
        """
        Extract manifolds for all layers.
        
        Args:
            layer_activations: Dict mapping layer_idx to (forget_acts, retain_acts)
            refusal_directions: Dict mapping layer_idx to refusal direction
        """
        for layer_idx in self.layer_indices:
            forget_acts, retain_acts = layer_activations[layer_idx]
            refusal_dir = refusal_directions[layer_idx]
            
            self.trainers[layer_idx].phase1_extract_manifolds(
                forget_acts, retain_acts, refusal_dir
            )
    
    def phase2_train(
        self,
        layer_inputs: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ):
        """
        Train all layer GMF modules.
        
        Args:
            layer_inputs: Dict mapping layer_idx to (forget_inputs, retain_inputs)
        """
        for layer_idx in self.layer_indices:
            forget_inputs, retain_inputs = layer_inputs[layer_idx]
            
            logger.info(f"Training GMF for layer {layer_idx}")
            self.trainers[layer_idx].phase2_train(forget_inputs, retain_inputs)
    
    def get_all_flow_weights(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Get weights for all layers."""
        return {
            layer_idx: trainer.get_flow_weights()
            for layer_idx, trainer in self.trainers.items()
        }