# Copyright (c) Steering Unlearn Project
"""
Trainer for Steering Unlearn Method - Round 5

Dual-channel steering with:
- Explicit steering direction
- Beta network for main update strength
- Small LoRA correction
- Direction consistency loss
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from itertools import chain
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from steering_unlearn.training.steering_module import (
    DualChannelSteeringModule,
    create_steering_modules,
    save_steering_modules,
    load_steering_modules,
)
from steering_unlearn.training.losses import SteeringLoss, LossComponents
from steering_unlearn.preprocessing.cone_target import ConeTarget


@dataclass
class TrainingConfig:
    """Configuration for steering training - Round 5."""
    num_epochs: int = 20
    lr: float = 0.01
    batch_size: int = 16
    scheduler_gamma: float = 0.9
    lambda_attractor: float = 0.5
    lambda_risk: float = 1.0
    lambda_retain: float = 2.0
    lambda_retain_kl: float = 1.0
    lambda_direction: float = 0.5  # Round 5: NEW
    kl_temperature: float = 1.0
    primary_target: str = "ignorance"
    steering_rank: int = 64
    gradient_clip: float = 1.0
    log_interval: int = 10
    save_interval: int = 5
    forget_batch_size: int = 4
    retain_batch_size: int = 8
    warmup_main_only_ratio: float = 0.0
    main_layer_idx: int = 19
    alpha_per_layer: Optional[Dict[int, float]] = None
    risk_threshold: float = 0.10
    delta_norm_clip: float = 1.0
    layer_loss_weights: Optional[Dict[int, float]] = None
    module_type: str = "dual_channel"  # Round 5: NEW
    beta_hidden_dim: int = 128  # Round 5: NEW


@dataclass
class TrainingMetrics:
    """Metrics logged during training - Round 5."""
    epoch: int
    total_loss: float
    attractor_loss: float
    risk_loss: float
    retain_loss: float
    direction_loss: float  # Round 5: NEW
    lr: float
    # Per-layer metrics
    attractor_per_layer: Dict[str, float]
    risk_per_layer: Dict[str, float]
    retain_per_layer: Dict[str, float]
    direction_per_layer: Dict[str, float]  # Round 5: NEW
    # Monitoring
    forget_gate_mean: float
    retain_gate_mean: float
    forget_risk_before: float
    forget_risk_after: float
    retain_risk_before: float
    retain_risk_after: float
    delta_norm_mean: float
    update_norm_mean: float
    risk_reduction: float
    forget_update_norm: float
    retain_update_norm: float
    direction_cosine_sim: float  # Round 5: NEW
    beta_mean: float  # Round 5: NEW - average beta value


class SteeringDataset(Dataset):
    """Dataset for steering training."""
    
    def __init__(
        self,
        forget_activations: Dict[int, torch.Tensor],
        retain_activations: Dict[int, torch.Tensor],
    ):
        self.forget_activations = forget_activations
        self.retain_activations = retain_activations
        self.n_forget = next(iter(forget_activations.values())).shape[0]
        self.n_retain = next(iter(retain_activations.values())).shape[0]
        self.layer_indices = list(forget_activations.keys())
        
    def __len__(self):
        return max(self.n_forget, self.n_retain)
    
    def __getitem__(self, idx):
        forget_idx = idx % self.n_forget
        forget_data = {
            layer: self.forget_activations[layer][forget_idx]
            for layer in self.layer_indices
        }
        retain_idx = idx % self.n_retain
        retain_data = {
            layer: self.retain_activations[layer][retain_idx]
            for layer in self.layer_indices
        }
        return {'forget': forget_data, 'retain': retain_data}


def steering_collate_fn(batch):
    """Collate function for steering dataset."""
    layer_indices = list(batch[0]['forget'].keys())
    forget_batch = {
        layer: torch.stack([item['forget'][layer] for item in batch])
        for layer in layer_indices
    }
    retain_batch = {
        layer: torch.stack([item['retain'][layer] for item in batch])
        for layer in layer_indices
    }
    return {'forget': forget_batch, 'retain': retain_batch}


class SteeringTrainer:
    """Trainer for steering modules - Round 5 with dual-channel steering."""
    
    def __init__(
        self,
        steering_modules: Dict[int, nn.Module],
        risk_probes: Dict[int, nn.Module],
        cone_targets: Dict[int, Dict[str, ConeTarget]],
        steering_directions: Dict[int, torch.Tensor],  # Round 5: NEW
        layer_indices: List[int],
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.steering_modules = steering_modules
        self.risk_probes = risk_probes
        self.cone_targets = cone_targets
        self.steering_directions = steering_directions
        self.layer_indices = layer_indices
        self.config = config
        self.device = device
        
        # Freeze all risk probes
        for probe in self.risk_probes.values():
            probe.eval()
            for param in probe.parameters():
                param.requires_grad = False
        
        # Move cone targets to device
        for layer_idx in layer_indices:
            for target_type in ['ignorance', 'refusal']:
                if hasattr(self.cone_targets[layer_idx][target_type], 'to'):
                    self.cone_targets[layer_idx][target_type].to(device)
        
        # Move steering directions to device
        for layer_idx in layer_indices:
            self.steering_directions[layer_idx] = self.steering_directions[layer_idx].to(device)
        
        # Loss function - Round 5
        self.loss_fn = SteeringLoss(
            lambda_attractor=config.lambda_attractor,
            lambda_risk=config.lambda_risk,
            lambda_retain=config.lambda_retain,
            lambda_retain_kl=getattr(config, 'lambda_retain_kl', 1.0),
            lambda_direction=getattr(config, 'lambda_direction', 0.5),
            kl_temperature=getattr(config, 'kl_temperature', 1.0),
            primary_target=config.primary_target,
        )
        
        # Optimizer
        all_params = list(chain(*[
            module.parameters() for module in self.steering_modules.values()
        ]))
        self.optimizer = optim.AdamW(all_params, lr=config.lr)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.scheduler_gamma
        )
        
        # Training history
        self.history: List[TrainingMetrics] = []
    
    def compute_detailed_metrics(
        self,
        forget_orig: Dict[int, torch.Tensor],
        forget_steered: Dict[int, torch.Tensor],
        forget_risk: Dict[int, torch.Tensor],
        retain_orig: Dict[int, torch.Tensor],
        retain_steered: Dict[int, torch.Tensor],
        retain_risk: Dict[int, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute detailed monitoring metrics for debugging."""
        metrics = {
            'forget_gate_mean': 0.0,
            'retain_gate_mean': 0.0,
            'forget_risk_before': 0.0,
            'forget_risk_after': 0.0,
            'retain_risk_before': 0.0,
            'retain_risk_after': 0.0,
            'delta_norm_mean': 0.0,
            'update_norm_mean': 0.0,
            'beta_mean': 0.0,  # Round 5: NEW
        }
        
        n_layers = len(self.layer_indices)
        
        for layer_idx in self.layer_indices:
            module = self.steering_modules[layer_idx]
            probe = self.risk_probes[layer_idx]
            
            # Compute gate values
            forget_gate = module.compute_gate(forget_risk[layer_idx])
            retain_gate = module.compute_gate(retain_risk[layer_idx])
            
            metrics['forget_gate_mean'] += forget_gate.mean().item()
            metrics['retain_gate_mean'] += retain_gate.mean().item()
            
            # Compute beta (Round 5: NEW)
            if hasattr(module, 'compute_beta'):
                beta = module.compute_beta(forget_orig[layer_idx].to(self.device))
                metrics['beta_mean'] += beta.mean().item()
            
            # Compute risk before and after steering
            with torch.no_grad():
                forget_risk_before = probe(forget_orig[layer_idx].float().to(self.device)).squeeze(-1)
                retain_risk_before = probe(retain_orig[layer_idx].float().to(self.device)).squeeze(-1)
                forget_risk_after = probe(forget_steered[layer_idx].float()).squeeze(-1)
                retain_risk_after = probe(retain_steered[layer_idx].float()).squeeze(-1)
            
            metrics['forget_risk_before'] += forget_risk_before.mean().item()
            metrics['forget_risk_after'] += forget_risk_after.mean().item()
            metrics['retain_risk_before'] += retain_risk_before.mean().item()
            metrics['retain_risk_after'] += retain_risk_after.mean().item()
            
            # Compute delta norm
            delta = forget_steered[layer_idx] - forget_orig[layer_idx].to(self.device)
            delta_norm = delta.norm(dim=-1).mean().item()
            metrics['delta_norm_mean'] += delta_norm
            
            # Compute final update norm
            alpha = module.alpha.item()
            gate = forget_gate
            update_norm = (alpha * gate * delta).norm(dim=-1).mean().item()
            metrics['update_norm_mean'] += update_norm
        
        # Average over layers
        for key in metrics:
            metrics[key] /= n_layers
        
        return metrics
    
    def apply_steering(
        self,
        activations: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Apply steering to activations."""
        steered = {}
        risk_scores = {}
        
        for layer_idx in self.layer_indices:
            act = activations[layer_idx].to(self.device)
            module = self.steering_modules[layer_idx]
            probe = self.risk_probes[layer_idx]
            
            # Compute risk score (for gating)
            with torch.no_grad():
                risk = probe(act.float()).squeeze(-1)
            risk_scores[layer_idx] = risk
            
            # Apply steering
            steered[layer_idx] = module.get_steered_activation(act, risk)
        
        return steered, risk_scores
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> TrainingMetrics:
        """Train for one epoch with detailed monitoring."""
        for module in self.steering_modules.values():
            module.train()
        
        # Accumulators
        total_loss_sum = 0.0
        attractor_loss_sum = 0.0
        risk_loss_sum = 0.0
        retain_loss_sum = 0.0
        direction_loss_sum = 0.0
        attractor_per_layer_sum = {f"layer{idx}": 0.0 for idx in self.layer_indices}
        risk_per_layer_sum = {f"layer{idx}": 0.0 for idx in self.layer_indices}
        retain_per_layer_sum = {f"layer{idx}": 0.0 for idx in self.layer_indices}
        direction_per_layer_sum = {f"layer{idx}": 0.0 for idx in self.layer_indices}
        
        forget_gate_sum = 0.0
        retain_gate_sum = 0.0
        forget_risk_before_sum = 0.0
        forget_risk_after_sum = 0.0
        retain_risk_before_sum = 0.0
        retain_risk_after_sum = 0.0
        delta_norm_sum = 0.0
        update_norm_sum = 0.0
        risk_reduction_sum = 0.0
        forget_update_norm_sum = 0.0
        retain_update_norm_sum = 0.0
        direction_cosine_sim_sum = 0.0
        beta_mean_sum = 0.0
        
        n_batches = 0
        n_metric_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            forget_orig = batch['forget']
            retain_orig = batch['retain']
            
            # Apply steering
            forget_steered, forget_risk = self.apply_steering(forget_orig)
            retain_steered, retain_risk = self.apply_steering(retain_orig)
            
            # Move to device
            retain_orig_device = {k: v.to(self.device) for k, v in retain_orig.items()}
            forget_orig_device = {k: v.to(self.device) for k, v in forget_orig.items()}
            
            # Compute loss - Round 5: Pass steering_directions
            loss_components = self.loss_fn(
                forget_original_activations=forget_orig_device,
                forget_steered_activations=forget_steered,
                retain_original_activations=retain_orig_device,
                retain_steered_activations=retain_steered,
                target_cones=self.cone_targets,
                risk_probes=self.risk_probes,
                layer_indices=self.layer_indices,
                steering_directions=self.steering_directions,
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_components.total_loss.backward()
            
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    chain(*[m.parameters() for m in self.steering_modules.values()]),
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss_sum += loss_components.total_loss.item()
            attractor_loss_sum += loss_components.attractor_loss.item()
            risk_loss_sum += loss_components.risk_loss.item()
            retain_loss_sum += loss_components.retain_loss.item()
            direction_loss_sum += loss_components.direction_loss.item()
            
            for idx in self.layer_indices:
                attractor_per_layer_sum[f"layer{idx}"] += loss_components.attractor_per_layer[idx]
                risk_per_layer_sum[f"layer{idx}"] += loss_components.risk_per_layer[idx]
                retain_per_layer_sum[f"layer{idx}"] += loss_components.retain_per_layer[idx]
                direction_per_layer_sum[f"layer{idx}"] += loss_components.direction_per_layer.get(idx, 0.0)
            
            # Compute detailed metrics
            if batch_idx % 5 == 0:
                detailed = self.compute_detailed_metrics(
                    forget_orig, forget_steered, forget_risk,
                    retain_orig, retain_steered, retain_risk,
                )
                forget_gate_sum += detailed['forget_gate_mean']
                retain_gate_sum += detailed['retain_gate_mean']
                forget_risk_before_sum += detailed['forget_risk_before']
                forget_risk_after_sum += detailed['forget_risk_after']
                retain_risk_before_sum += detailed['retain_risk_before']
                retain_risk_after_sum += detailed['retain_risk_after']
                delta_norm_sum += detailed['delta_norm_mean']
                update_norm_sum += detailed['update_norm_mean']
                beta_mean_sum += detailed['beta_mean']
                
                risk_reduction_sum += loss_components.risk_reduction
                forget_update_norm_sum += loss_components.forget_update_norm
                retain_update_norm_sum += loss_components.retain_update_norm
                direction_cosine_sim_sum += loss_components.direction_cosine_sim
                
                n_metric_batches += 1
            
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss_components.total_loss.item():.4f}",
                'attr': f"{loss_components.attractor_loss.item():.4f}",
                'risk': f"{loss_components.risk_loss.item():.4f}",
                'dir': f"{loss_components.direction_loss.item():.4f}",
            })
        
        n_metric_batches = max(1, n_metric_batches)
        
        metrics = TrainingMetrics(
            epoch=epoch,
            total_loss=total_loss_sum / n_batches,
            attractor_loss=attractor_loss_sum / n_batches,
            risk_loss=risk_loss_sum / n_batches,
            retain_loss=retain_loss_sum / n_batches,
            direction_loss=direction_loss_sum / n_batches,
            lr=self.optimizer.param_groups[0]['lr'],
            attractor_per_layer={k: v / n_batches for k, v in attractor_per_layer_sum.items()},
            risk_per_layer={k: v / n_batches for k, v in risk_per_layer_sum.items()},
            retain_per_layer={k: v / n_batches for k, v in retain_per_layer_sum.items()},
            direction_per_layer={k: v / n_batches for k, v in direction_per_layer_sum.items()},
            forget_gate_mean=forget_gate_sum / n_metric_batches,
            retain_gate_mean=retain_gate_sum / n_metric_batches,
            forget_risk_before=forget_risk_before_sum / n_metric_batches,
            forget_risk_after=forget_risk_after_sum / n_metric_batches,
            retain_risk_before=retain_risk_before_sum / n_metric_batches,
            retain_risk_after=retain_risk_after_sum / n_metric_batches,
            delta_norm_mean=delta_norm_sum / n_metric_batches,
            update_norm_mean=update_norm_sum / n_metric_batches,
            risk_reduction=risk_reduction_sum / n_metric_batches,
            forget_update_norm=forget_update_norm_sum / n_metric_batches,
            retain_update_norm=retain_update_norm_sum / n_metric_batches,
            direction_cosine_sim=direction_cosine_sim_sum / n_metric_batches,
            beta_mean=beta_mean_sum / n_metric_batches,
        )
        
        self.scheduler.step()
        
        return metrics
    
    def train(
        self,
        forget_activations: Dict[int, torch.Tensor],
        retain_activations: Dict[int, torch.Tensor],
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> List[TrainingMetrics]:
        """Run the full training loop with detailed monitoring."""
        dataset = SteeringDataset(forget_activations, retain_activations)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=steering_collate_fn,
            num_workers=0,
        )
        
        print(f"\nStarting training with {len(dataset)} samples")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Number of batches: {len(dataloader)}")
        print(f"Layer indices: {self.layer_indices}")
        print(f"Risk threshold: {self.config.risk_threshold}")
        print(f"Alpha per layer: {self.config.alpha_per_layer}")
        print(f"Module type: {self.config.module_type}")
        
        for epoch in range(self.config.num_epochs):
            metrics = self.train_epoch(dataloader, epoch)
            self.history.append(metrics)
            
            if verbose:
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs} Summary:")
                print(f"  Total Loss: {metrics.total_loss:.4f}")
                print(f"  Attractor Loss: {metrics.attractor_loss:.4f}")
                print(f"  Risk Loss: {metrics.risk_loss:.4f}")
                print(f"  Retain Loss: {metrics.retain_loss:.4f}")
                print(f"  Direction Loss: {metrics.direction_loss:.4f}")
                print(f"  Learning Rate: {metrics.lr:.6f}")
                print(f"  --- Detailed Monitoring ---")
                print(f"  Forget Gate Mean: {metrics.forget_gate_mean:.4f}")
                print(f"  Retain Gate Mean: {metrics.retain_gate_mean:.4f}")
                print(f"  Forget Risk Before: {metrics.forget_risk_before:.4f}")
                print(f"  Forget Risk After: {metrics.forget_risk_after:.4f}")
                print(f"  Retain Risk Before: {metrics.retain_risk_before:.4f}")
                print(f"  Retain Risk After: {metrics.retain_risk_after:.4f}")
                print(f"  Delta Norm Mean: {metrics.delta_norm_mean:.4f}")
                print(f"  Update Norm Mean: {metrics.update_norm_mean:.4f}")
                print(f"  Risk Reduction: {metrics.risk_reduction:.4f}")
                print(f"  Forget Update Norm: {metrics.forget_update_norm:.4f}")
                print(f"  Retain Update Norm: {metrics.retain_update_norm:.4f}")
                print(f"  Direction Cosine Sim: {metrics.direction_cosine_sim:.4f}")
                print(f"  Beta Mean: {metrics.beta_mean:.4f}")
            
            if save_dir and (epoch + 1) % self.config.save_interval == 0:
                checkpoint_dir = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}")
                save_steering_modules(self.steering_modules, checkpoint_dir)
                print(f"  Saved checkpoint to {checkpoint_dir}")
        
        if save_dir:
            final_dir = os.path.join(save_dir, "final")
            save_steering_modules(self.steering_modules, final_dir)
            
            history_path = os.path.join(save_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump([asdict(m) for m in self.history], f, indent=2)
            print(f"\nSaved final model and history to {save_dir}")
        
        return self.history
    
    def get_steered_activations(
        self,
        activations: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Get steered activations without computing loss."""
        for module in self.steering_modules.values():
            module.eval()
        steered, _ = self.apply_steering(activations)
        return steered


def train_steering_modules(
    all_activations: Dict[str, Dict[int, torch.Tensor]],
    risk_probes: Dict[int, nn.Module],
    cone_targets: Dict[int, Dict[str, ConeTarget]],
    steering_directions: Dict[int, torch.Tensor],
    config: TrainingConfig,
    device: str = "cuda",
    save_dir: Optional[str] = None,
) -> Tuple[Dict[int, nn.Module], List[TrainingMetrics]]:
    """Train steering modules for all target layers - Round 5."""
    layer_indices = list(risk_probes.keys())
    hidden_dim = next(iter(all_activations['forget'].values())).shape[1]
    
    print(f"\n{'='*60}")
    print("Training Steering Modules - Round 5 Dual-Channel")
    print(f"{'='*60}")
    print(f"Layers: {layer_indices}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Device: {device}")
    
    # Create steering modules
    steering_modules = create_steering_modules(
        layer_indices=layer_indices,
        hidden_dim=hidden_dim,
        steering_directions=steering_directions,
        module_type=getattr(config, 'module_type', 'dual_channel'),
        rank=config.steering_rank,
        device=device,
        alpha_per_layer=config.alpha_per_layer,
        risk_threshold=config.risk_threshold,
        delta_norm_clip=config.delta_norm_clip,
        beta_hidden_dim=getattr(config, 'beta_hidden_dim', 128),
    )
    
    # Count parameters
    total_params = 0
    for layer_idx, module in steering_modules.items():
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  Layer {layer_idx}: {n_params:,} trainable parameters")
        total_params += n_params
    print(f"Total trainable parameters: {total_params:,}")
    
    # Create trainer
    trainer = SteeringTrainer(
        steering_modules=steering_modules,
        risk_probes=risk_probes,
        cone_targets=cone_targets,
        steering_directions=steering_directions,
        layer_indices=layer_indices,
        config=config,
        device=device,
    )
    
    # Train
    history = trainer.train(
        forget_activations=all_activations['forget'],
        retain_activations=all_activations['retain'],
        save_dir=save_dir,
        verbose=True,
    )
    
    return steering_modules, history