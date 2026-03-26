# Copyright (c) Steering Unlearn Project
"""
Cone Target Constructor for Steering Unlearn Method

This module constructs ignorance and refusal target cones from reference activations.
Each cone contains:
- axis: the mean direction of normalized reference activations
- membership_threshold: cosine similarity threshold (80th percentile)
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ConeTarget:
    """Represents a target cone with axis and membership threshold."""
    axis: torch.Tensor  # Normalized mean direction [hidden_dim]
    membership_threshold: float  # Cosine similarity threshold
    mean_similarity: float  # Mean cosine similarity
    percentile_stats: Dict  # Statistics about similarity distribution
    
    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            'axis': self.axis.cpu().numpy().tolist(),
            'membership_threshold': self.membership_threshold,
            'mean_similarity': self.mean_similarity,
            'percentile_stats': self.percentile_stats,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConeTarget':
        """Create from dictionary."""
        return cls(
            axis=torch.tensor(data['axis']),
            membership_threshold=data['membership_threshold'],
            mean_similarity=data['mean_similarity'],
            percentile_stats=data['percentile_stats'],
        )
    
    def to(self, device):
        """Move cone to specified device."""
        self.axis = self.axis.to(device)
        return self


def normalize_activations(activations: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize activations along the last dimension.
    
    Args:
        activations: Tensor of shape [num_samples, hidden_dim]
        
    Returns:
        Normalized tensor of same shape
    """
    # Convert to float32 for numerical stability
    activations = activations.float()
    norms = torch.norm(activations, p=2, dim=-1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)  # Avoid division by zero
    return activations / norms


def compute_cosine_similarities(
    activations: torch.Tensor,
    axis: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarities between activations and axis.
    
    Args:
        activations: Normalized activations [num_samples, hidden_dim]
        axis: Normalized axis direction [hidden_dim]
        
    Returns:
        Tensor of cosine similarities [num_samples]
    """
    # Ensure both are normalized
    activations_norm = normalize_activations(activations)
    axis_norm = axis / torch.norm(axis, p=2)
    
    # Compute cosine similarity
    similarities = torch.matmul(activations_norm, axis_norm)
    return similarities


def construct_cone_target(
    reference_activations: torch.Tensor,
    percentile: float = 0.80,
    verbose: bool = True,
) -> ConeTarget:
    """
    Construct a cone target from reference activations.
    
    Args:
        reference_activations: Tensor of shape [num_samples, hidden_dim]
        percentile: Percentile for membership threshold (default: 0.80)
        verbose: Whether to print statistics
        
    Returns:
        ConeTarget object with axis and threshold
    """
    # Step 1: Normalize all reference activations
    normalized_acts = normalize_activations(reference_activations)
    
    # Step 2: Compute mean direction (axis)
    axis = torch.mean(normalized_acts, dim=0)
    # Normalize the axis
    axis = axis / torch.norm(axis, p=2)
    
    # Step 3: Compute cosine similarities with axis
    similarities = compute_cosine_similarities(reference_activations, axis)
    
    # Step 4: Compute statistics
    similarities_np = similarities.cpu().numpy()
    mean_sim = float(np.mean(similarities_np))
    std_sim = float(np.std(similarities_np))
    
    # Compute percentile thresholds
    percentiles_to_compute = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    percentile_values = {}
    for p in percentiles_to_compute:
        percentile_values[f'p{int(p*100)}'] = float(np.percentile(similarities_np, p * 100))
    
    # Get the membership threshold at specified percentile
    membership_threshold = float(np.percentile(similarities_np, percentile * 100))
    
    percentile_stats = {
        'mean': mean_sim,
        'std': std_sim,
        'min': float(np.min(similarities_np)),
        'max': float(np.max(similarities_np)),
        'percentiles': percentile_values,
    }
    
    if verbose:
        print(f"  Cone axis norm: {torch.norm(axis).item():.4f}")
        print(f"  Mean cosine similarity: {mean_sim:.4f}")
        print(f"  Std cosine similarity: {std_sim:.4f}")
        print(f"  Membership threshold (p{int(percentile*100)}): {membership_threshold:.4f}")
    
    return ConeTarget(
        axis=axis,
        membership_threshold=membership_threshold,
        mean_similarity=mean_sim,
        percentile_stats=percentile_stats,
    )


def construct_all_cone_targets(
    all_activations: Dict[str, Dict[int, torch.Tensor]],
    layer_indices: list,
    percentile: float = 0.80,
    save_dir: Optional[str] = None,
) -> Dict[int, Dict[str, ConeTarget]]:
    """
    Construct ignorance and refusal cone targets for all layers.
    
    Args:
        all_activations: Dictionary with activations organized by data type and layer
        layer_indices: List of layer indices
        percentile: Percentile for membership threshold
        save_dir: Directory to save cone targets
        
    Returns:
        Dictionary with cone targets organized by layer and type
    """
    cone_targets = {}
    
    for layer_idx in layer_indices:
        print(f"\n{'='*50}")
        print(f"Constructing Cone Targets for Layer {layer_idx}")
        print(f"{'='*50}")
        
        layer_targets = {}
        
        # Construct ignorance cone
        print("\nConstructing Ignorance Cone:")
        ignorance_acts = all_activations['ignorance'][layer_idx]
        ignorance_cone = construct_cone_target(
            reference_activations=ignorance_acts,
            percentile=percentile,
            verbose=True,
        )
        layer_targets['ignorance'] = ignorance_cone
        
        # Construct refusal cone
        print("\nConstructing Refusal Cone:")
        refusal_acts = all_activations['refusal'][layer_idx]
        refusal_cone = construct_cone_target(
            reference_activations=refusal_acts,
            percentile=percentile,
            verbose=True,
        )
        layer_targets['refusal'] = refusal_cone
        
        cone_targets[layer_idx] = layer_targets
        
        # Save to disk if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save ignorance cone
            ignorance_path = os.path.join(save_dir, f"ignorance_cone_layer{layer_idx}.pt")
            torch.save(ignorance_cone.to_dict(), ignorance_path)
            print(f"Saved ignorance cone to {ignorance_path}")
            
            # Save refusal cone
            refusal_path = os.path.join(save_dir, f"refusal_cone_layer{layer_idx}.pt")
            torch.save(refusal_cone.to_dict(), refusal_path)
            print(f"Saved refusal cone to {refusal_path}")
    
    # Save combined file
    if save_dir:
        combined_path = os.path.join(save_dir, "all_cone_targets.pt")
        combined_data = {}
        for layer_idx, targets in cone_targets.items():
            combined_data[f"layer{layer_idx}"] = {
                'ignorance': targets['ignorance'].to_dict(),
                'refusal': targets['refusal'].to_dict(),
            }
        torch.save(combined_data, combined_path)
        print(f"\nSaved combined cone targets to {combined_path}")
    
    return cone_targets


def load_cone_targets(
    save_dir: str,
    layer_indices: list,
) -> Dict[int, Dict[str, ConeTarget]]:
    """
    Load cone targets from disk.
    
    Args:
        save_dir: Directory containing saved cone targets
        layer_indices: List of layer indices
        
    Returns:
        Dictionary with cone targets organized by layer and type
    """
    cone_targets = {}
    
    for layer_idx in layer_indices:
        layer_targets = {}
        
        # Load ignorance cone
        ignorance_path = os.path.join(save_dir, f"ignorance_cone_layer{layer_idx}.pt")
        ignorance_data = torch.load(ignorance_path)
        layer_targets['ignorance'] = ConeTarget.from_dict(ignorance_data)
        
        # Load refusal cone
        refusal_path = os.path.join(save_dir, f"refusal_cone_layer{layer_idx}.pt")
        refusal_data = torch.load(refusal_path)
        layer_targets['refusal'] = ConeTarget.from_dict(refusal_data)
        
        cone_targets[layer_idx] = layer_targets
    
    return cone_targets


def check_membership(
    activation: torch.Tensor,
    cone: ConeTarget,
) -> torch.Tensor:
    """
    Check if activation is within the cone (above membership threshold).
    
    Args:
        activation: Single activation [hidden_dim] or batch [batch, hidden_dim]
        cone: ConeTarget object
        
    Returns:
        Boolean tensor indicating membership
    """
    if activation.dim() == 1:
        activation = activation.unsqueeze(0)
    
    similarities = compute_cosine_similarities(activation, cone.axis)
    return similarities >= cone.membership_threshold


def get_distance_to_cone(
    activation: torch.Tensor,
    cone: ConeTarget,
) -> torch.Tensor:
    """
    Compute distance of activation to cone (negative cosine similarity).
    Lower is closer to cone.
    
    Args:
        activation: Activation tensor [batch, hidden_dim]
        cone: ConeTarget object
        
    Returns:
        Distance tensor [batch]
    """
    similarities = compute_cosine_similarities(activation, cone.axis)
    return 1.0 - similarities  # Distance = 1 - similarity


if __name__ == "__main__":
    # Test the cone target constructor
    print("Cone Target Constructor Module")
    
    # Create dummy activations
    num_samples = 100
    hidden_dim = 4096
    
    # Random activations with some structure
    base_direction = torch.randn(hidden_dim)
    base_direction = base_direction / torch.norm(base_direction)
    
    activations = []
    for _ in range(num_samples):
        noise = torch.randn(hidden_dim) * 0.5
        act = base_direction + noise
        activations.append(act)
    activations = torch.stack(activations)
    
    # Construct cone
    cone = construct_cone_target(activations, percentile=0.80, verbose=True)
    
    print(f"\nCone axis shape: {cone.axis.shape}")
    print(f"Membership threshold: {cone.membership_threshold:.4f}")