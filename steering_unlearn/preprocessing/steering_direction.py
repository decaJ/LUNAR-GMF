# Copyright (c) Steering Unlearn Project
"""
Steering Direction Computation - Round 5

Compute explicit steering direction for each layer:
    v_l = Normalize(c_ign - mu_f)

where:
    - c_ign: ignorance cone axis
    - mu_f: forget activation mean
"""

import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_steering_directions(
    forget_activations: Dict[int, torch.Tensor],
    cone_targets: Dict[int, Dict],
    layer_indices: List[int],
    save_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """
    Compute explicit steering direction for each layer.
    
    Direction = Normalize(ignorance_cone_axis - forget_mean)
    
    This provides the main "push" direction that moves forget samples
    toward the ignorance region.
    
    Args:
        forget_activations: Dict mapping layer_idx to forget activations [N, hidden_dim]
        cone_targets: Dict mapping layer_idx to cone targets (ignorance/refusal)
        layer_indices: List of layer indices
        save_dir: Directory to save steering directions
        device: Device to use
        
    Returns:
        Dict mapping layer_idx to steering direction [hidden_dim]
    """
    steering_directions = {}
    
    print("\n" + "="*60)
    print("Computing Steering Directions")
    print("="*60)
    
    for layer_idx in layer_indices:
        # Get forget activations
        forget_acts = forget_activations[layer_idx].to(device)  # [N, hidden_dim]
        
        # Compute forget mean
        forget_mean = forget_acts.mean(dim=0)  # [hidden_dim]
        
        # Get ignorance cone axis
        cone = cone_targets[layer_idx]['ignorance']
        if hasattr(cone, 'axis'):
            ignorance_axis = cone.axis.to(device)
        else:
            ignorance_axis = torch.tensor(cone['axis']).to(device)
        
        # Compute steering direction: v = Normalize(c_ign - mu_f)
        direction = ignorance_axis.float() - forget_mean.float()
        direction = F.normalize(direction, dim=0)
        
        steering_directions[layer_idx] = direction.cpu()  # Save to CPU for storage
        
        # Compute statistics
        cos_sim = F.cosine_similarity(
            forget_mean.unsqueeze(0).float(),
            ignorance_axis.unsqueeze(0).float()
        ).item()
        
        direction_norm = direction.norm().item()
        forget_mean_norm = forget_mean.norm().item()
        ignorance_axis_norm = ignorance_axis.norm().item()
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Forget mean norm: {forget_mean_norm:.4f}")
        print(f"  Ignorance axis norm: {ignorance_axis_norm:.4f}")
        print(f"  Cosine(forget_mean, ignorance_axis): {cos_sim:.4f}")
        print(f"  Steering direction norm: {direction_norm:.4f}")
        
        # Save to disk
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            direction_path = os.path.join(save_dir, f"steering_direction_layer{layer_idx}.pt")
            torch.save({
                'layer_idx': layer_idx,
                'direction': direction.cpu(),
                'forget_mean': forget_mean.cpu(),
                'ignorance_axis': ignorance_axis.cpu() if hasattr(cone, 'axis') else torch.tensor(cone['axis']),
                'cosine_similarity': cos_sim,
            }, direction_path)
            print(f"  Saved to {direction_path}")
    
    # Save combined directions
    if save_dir:
        combined_path = os.path.join(save_dir, "all_steering_directions.pt")
        torch.save({
            'layer_indices': layer_indices,
            'directions': {f"layer{idx}": d for idx, d in steering_directions.items()},
        }, combined_path)
        print(f"\nSaved combined directions to {combined_path}")
    
    return steering_directions


def load_steering_directions(
    save_dir: str,
    layer_indices: List[int],
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """
    Load steering directions from disk.
    
    Args:
        save_dir: Directory containing saved directions
        layer_indices: List of layer indices
        device: Device to load to
        
    Returns:
        Dict mapping layer_idx to steering direction [hidden_dim]
    """
    directions = {}
    
    for layer_idx in layer_indices:
        direction_path = os.path.join(save_dir, f"steering_direction_layer{layer_idx}.pt")
        checkpoint = torch.load(direction_path, map_location=device)
        directions[layer_idx] = checkpoint['direction'].to(device)
        print(f"Loaded steering direction for layer {layer_idx}")
    
    return directions


if __name__ == "__main__":
    # Test steering direction computation
    print("Testing Steering Direction Computation")
    
    # Create dummy data
    batch_size = 100
    hidden_dim = 4096
    layer_indices = [18, 19, 21]
    
    forget_activations = {
        idx: torch.randn(batch_size, hidden_dim) + 0.5  # Shifted mean
        for idx in layer_indices
    }
    
    # Create dummy cone targets
    cone_targets = {}
    for idx in layer_indices:
        # Ignorance axis pointing in a different direction
        ignorance_axis = torch.randn(hidden_dim)
        ignorance_axis = F.normalize(ignorance_axis, dim=0)
        
        cone_targets[idx] = {
            'ignorance': {'axis': ignorance_axis},
            'refusal': {'axis': torch.randn(hidden_dim)},
        }
    
    # Compute steering directions
    directions = compute_steering_directions(
        forget_activations=forget_activations,
        cone_targets=cone_targets,
        layer_indices=layer_indices,
        save_dir=None,
    )
    
    print("\nSteering directions computed successfully!")
    for idx, d in directions.items():
        print(f"  Layer {idx}: direction shape = {d.shape}, norm = {d.norm().item():.4f}")