# Copyright (c) Steering Unlearn Project
"""
Risk Probe Trainer for Steering Unlearn Method

This module trains risk probes for each steering layer.
A risk probe estimates whether an activation still retains forget knowledge recoverability.

Input: layer activation
Output: risk score in [0, 1] (1 = forget knowledge present, 0 = no forget knowledge)
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, accuracy_score


class RiskProbeMLP(nn.Module):
    """
    Lightweight MLP probe for risk estimation.
    
    Architecture:
        Input: [batch, hidden_dim]
        Linear -> ReLU -> Linear -> Sigmoid
        Output: [batch, 1] (risk score in [0, 1])
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input activations [batch, hidden_dim]
            
        Returns:
            Risk scores [batch, 1]
        """
        return self.layers(x)
    
    def get_risk_score(self, x: torch.Tensor) -> torch.Tensor:
        """Get risk score, returns [batch] tensor."""
        return self.forward(x).squeeze(-1)


class RiskProbeLinear(nn.Module):
    """
    Linear probe for risk estimation (simpler alternative).
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.input_dim = input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))
    
    def get_risk_score(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).squeeze(-1)


@dataclass
class ProbeTrainingConfig:
    """Configuration for probe training."""
    hidden_dim: int = 256
    epochs: int = 50
    lr: float = 0.001
    batch_size: int = 32
    val_split: float = 0.2
    patience: int = 10  # Early stopping patience
    weight_decay: float = 0.01


@dataclass
class ProbeTrainingResult:
    """Results from probe training."""
    probe: nn.Module
    layer_idx: int
    train_loss_history: List[float]
    val_loss_history: List[float]
    train_accuracy: float
    val_accuracy: float
    auc_score: float
    forget_avg_risk: float
    retain_avg_risk: float


def prepare_probe_training_data(
    forget_activations: torch.Tensor,
    retain_activations: torch.Tensor,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare data loaders for probe training.
    
    Args:
        forget_activations: Activations from forget set [N_forget, hidden_dim]
        retain_activations: Activations from retain set [N_retain, hidden_dim]
        val_split: Fraction of data for validation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert to float32 for training stability
    forget_activations = forget_activations.float()
    retain_activations = retain_activations.float()
    
    # Create labels: forget=1, retain=0
    n_forget = forget_activations.shape[0]
    n_retain = retain_activations.shape[0]
    
    forget_labels = torch.ones(n_forget, 1)
    retain_labels = torch.zeros(n_retain, 1)
    
    # Combine data
    all_activations = torch.cat([forget_activations, retain_activations], dim=0)
    all_labels = torch.cat([forget_labels, retain_labels], dim=0)
    
    # Shuffle indices
    n_total = all_activations.shape[0]
    indices = torch.randperm(n_total)
    
    # Split into train/val
    n_val = int(n_total * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Create datasets
    train_dataset = TensorDataset(
        all_activations[train_indices],
        all_labels[train_indices]
    )
    val_dataset = TensorDataset(
        all_activations[val_indices],
        all_labels[val_indices]
    )
    
    return train_dataset, val_dataset


def train_risk_probe(
    forget_activations: torch.Tensor,
    retain_activations: torch.Tensor,
    layer_idx: int,
    config: ProbeTrainingConfig,
    device: str = "cuda",
    use_linear: bool = False,
    verbose: bool = True,
) -> ProbeTrainingResult:
    """
    Train a risk probe for a specific layer.
    
    Args:
        forget_activations: Activations from forget set
        retain_activations: Activations from retain set
        layer_idx: Layer index (for logging)
        config: Training configuration
        device: Device to train on
        use_linear: Use linear probe instead of MLP
        verbose: Print training progress
        
    Returns:
        ProbeTrainingResult object
    """
    input_dim = forget_activations.shape[1]
    
    # Create probe
    if use_linear:
        probe = RiskProbeLinear(input_dim=input_dim)
    else:
        probe = RiskProbeMLP(input_dim=input_dim, hidden_dim=config.hidden_dim)
    probe = probe.to(device)
    
    # Prepare data
    train_dataset, val_dataset = prepare_probe_training_data(
        forget_activations=forget_activations.to(device),
        retain_activations=retain_activations.to(device),
        val_split=config.val_split,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        probe.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training loop
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_probe_state = None
    
    for epoch in range(config.epochs):
        # Training
        probe.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = probe(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)
        
        # Validation
        probe.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = probe(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)
        
        scheduler.step(val_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config.epochs}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_probe_state = probe.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best state
    if best_probe_state is not None:
        probe.load_state_dict(best_probe_state)
    
    # Compute final metrics
    probe.eval()
    
    # Get predictions for all data
    with torch.no_grad():
        # Training set predictions
        all_train_preds = []
        all_train_labels = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            preds = probe(batch_x).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(batch_y.numpy())
        
        # Validation set predictions
        all_val_preds = []
        all_val_labels = []
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            preds = probe(batch_x).cpu().numpy()
            all_val_preds.extend(preds)
            all_val_labels.extend(batch_y.numpy())
        
        # Compute metrics
        train_preds_binary = (np.array(all_train_preds) > 0.5).astype(int)
        train_labels = np.array(all_train_labels).astype(int)
        train_accuracy = accuracy_score(train_labels, train_preds_binary)
        
        val_preds_binary = (np.array(all_val_preds) > 0.5).astype(int)
        val_labels = np.array(all_val_labels).astype(int)
        val_accuracy = accuracy_score(val_labels, val_preds_binary)
        
        # AUC score (on combined data)
        all_preds = np.array(all_train_preds + all_val_preds).flatten()
        all_labels = np.array(all_train_labels + all_val_labels).flatten()
        auc_score = roc_auc_score(all_labels, all_preds)
        
        # Average risk scores - convert to float32
        forget_acts_device = forget_activations.float().to(device)
        retain_acts_device = retain_activations.float().to(device)
        
        with torch.no_grad():
            forget_risks = probe(forget_acts_device).cpu().numpy().flatten()
            retain_risks = probe(retain_acts_device).cpu().numpy().flatten()
        
        forget_avg_risk = float(np.mean(forget_risks))
        retain_avg_risk = float(np.mean(retain_risks))
    
    if verbose:
        print(f"\n  Final Metrics:")
        print(f"    Train Accuracy: {train_accuracy:.4f}")
        print(f"    Val Accuracy: {val_accuracy:.4f}")
        print(f"    AUC Score: {auc_score:.4f}")
        print(f"    Forget Avg Risk: {forget_avg_risk:.4f}")
        print(f"    Retain Avg Risk: {retain_avg_risk:.4f}")
    
    return ProbeTrainingResult(
        probe=probe,
        layer_idx=layer_idx,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        auc_score=auc_score,
        forget_avg_risk=forget_avg_risk,
        retain_avg_risk=retain_avg_risk,
    )


def train_all_risk_probes(
    all_activations: Dict[str, Dict[int, torch.Tensor]],
    layer_indices: List[int],
    config: ProbeTrainingConfig,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[int, ProbeTrainingResult]:
    """
    Train risk probes for all target layers.
    
    Args:
        all_activations: Dictionary with activations organized by data type and layer
        layer_indices: List of layer indices
        config: Training configuration
        device: Device to train on
        save_dir: Directory to save probes
        verbose: Print training progress
        
    Returns:
        Dictionary mapping layer index to ProbeTrainingResult
    """
    results = {}
    
    for layer_idx in layer_indices:
        print(f"\n{'='*50}")
        print(f"Training Risk Probe for Layer {layer_idx}")
        print(f"{'='*50}")
        
        forget_acts = all_activations['forget'][layer_idx]
        retain_acts = all_activations['retain'][layer_idx]
        
        result = train_risk_probe(
            forget_activations=forget_acts,
            retain_activations=retain_acts,
            layer_idx=layer_idx,
            config=config,
            device=device,
            use_linear=False,
            verbose=verbose,
        )
        
        results[layer_idx] = result
        
        # Save probe
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save probe weights
            probe_path = os.path.join(save_dir, f"risk_probe_layer{layer_idx}.pt")
            torch.save({
                'model_state_dict': result.probe.state_dict(),
                'input_dim': result.probe.input_dim,
                'hidden_dim': result.probe.hidden_dim if hasattr(result.probe, 'hidden_dim') else None,
                'layer_idx': layer_idx,
                'train_accuracy': result.train_accuracy,
                'val_accuracy': result.val_accuracy,
                'auc_score': result.auc_score,
                'forget_avg_risk': result.forget_avg_risk,
                'retain_avg_risk': result.retain_avg_risk,
            }, probe_path)
            print(f"  Saved probe to {probe_path}")
            
            # Save training log
            log_path = os.path.join(save_dir, f"risk_probe_layer{layer_idx}_log.json")
            log_data = {
                'layer_idx': layer_idx,
                'train_loss_history': result.train_loss_history,
                'val_loss_history': result.val_loss_history,
                'train_accuracy': result.train_accuracy,
                'val_accuracy': result.val_accuracy,
                'auc_score': result.auc_score,
                'forget_avg_risk': result.forget_avg_risk,
                'retain_avg_risk': result.retain_avg_risk,
            }
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
    
    # Save summary
    if save_dir:
        summary_path = os.path.join(save_dir, "probe_training_summary.json")
        summary = {}
        for layer_idx, result in results.items():
            summary[f"layer{layer_idx}"] = {
                'train_accuracy': result.train_accuracy,
                'val_accuracy': result.val_accuracy,
                'auc_score': result.auc_score,
                'forget_avg_risk': result.forget_avg_risk,
                'retain_avg_risk': result.retain_avg_risk,
            }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved probe training summary to {summary_path}")
    
    return results


def load_risk_probe(
    probe_path: str,
    device: str = "cuda",
) -> nn.Module:
    """
    Load a trained risk probe from disk.
    
    Args:
        probe_path: Path to saved probe
        device: Device to load to
        
    Returns:
        Loaded probe module
    """
    checkpoint = torch.load(probe_path, map_location=device)
    
    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint.get('hidden_dim', 256)
    
    # Create probe
    probe = RiskProbeMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    probe.load_state_dict(checkpoint['model_state_dict'])
    probe = probe.to(device)
    probe.eval()
    
    return probe


def load_all_risk_probes(
    save_dir: str,
    layer_indices: List[int],
    device: str = "cuda",
) -> Dict[int, nn.Module]:
    """
    Load all risk probes from disk.
    
    Args:
        save_dir: Directory containing saved probes
        layer_indices: List of layer indices
        device: Device to load to
        
    Returns:
        Dictionary mapping layer index to probe module
    """
    probes = {}
    
    for layer_idx in layer_indices:
        probe_path = os.path.join(save_dir, f"risk_probe_layer{layer_idx}.pt")
        probes[layer_idx] = load_risk_probe(probe_path, device)
        print(f"Loaded probe for layer {layer_idx}")
    
    return probes


if __name__ == "__main__":
    # Test the risk probe trainer
    print("Risk Probe Trainer Module")
    
    # Create dummy data
    n_forget = 200
    n_retain = 500
    hidden_dim = 4096
    
    # Forget activations (cluster around some direction)
    forget_center = torch.randn(hidden_dim)
    forget_acts = forget_center + torch.randn(n_forget, hidden_dim) * 0.3
    
    # Retain activations (different cluster)
    retain_center = torch.randn(hidden_dim)
    retain_acts = retain_center + torch.randn(n_retain, hidden_dim) * 0.3
    
    # Train config
    config = ProbeTrainingConfig(
        hidden_dim=256,
        epochs=50,
        lr=0.001,
        batch_size=32,
        val_split=0.2,
    )
    
    # Train probe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = train_risk_probe(
        forget_activations=forget_acts,
        retain_activations=retain_acts,
        layer_idx=0,
        config=config,
        device=device,
        verbose=True,
    )
    
    print(f"\nProbe trained successfully!")
    print(f"AUC Score: {result.auc_score:.4f}")