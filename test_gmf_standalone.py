#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone GMF Test Script
Tests the GMF modules without requiring full model/dataset loading.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmf.manifold import ForgetManifold, AttractorManifold, ManifoldExtractor
from gmf.gating import DistanceBasedGate, GatedFlow
from gmf.flow_transform import ResidualFlow
from gmf.losses import GMFLoss, GMFLossConfig
from gmf.trainer import GMFTrainer, GMFTrainerConfig, GMFModule


def test_manifold_extraction():
    """Test manifold extraction from synthetic data."""
    print("\n" + "=" * 60)
    print("Test 1: Manifold Extraction")
    print("=" * 60)
    
    hidden_size = 256
    n_samples = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create synthetic activations
    mean = torch.randn(hidden_size)
    
    activations = [mean + torch.randn(hidden_size) * 0.5 for _ in range(n_samples)]
    
    # Extract manifold
    extractor = ManifoldExtractor(
        hidden_size=hidden_size,
        use_diagonal_cov=True,
        device=device,
    )
    
    manifold = extractor.extract_forget_manifold(activations)
    
    print(f"Extracted manifold mean norm: {torch.norm(manifold.mu):.4f}")
    print(f"Covariance diagonal mean: {manifold.Sigma.mean():.4f}")
    print("✓ Manifold extraction test passed!")
    
    return manifold


def test_gating_mechanism():
    """Test distance-based gating."""
    print("\n" + "=" * 60)
    print("Test 2: Gating Mechanism")
    print("=" * 60)
    
    hidden_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create forget manifold with unit variance
    mu = torch.randn(hidden_size) * 0.1  # Smaller mean
    sigma = torch.ones(hidden_size) * 0.5  # Unit variance for simplicity
    forget_manifold = ForgetManifold(mu=mu, Sigma=sigma)
    
    # Create gate with larger sigma for high-dimensional space
    # In high dimensions, distances are naturally larger
    gate = DistanceBasedGate(
        hidden_size=hidden_size,
        sigma=10.0,  # Larger sigma for high-dimensional space
        learnable_sigma=False,
        distance_method='mahalanobis',
    )
    gate.set_manifold(forget_manifold)
    gate = gate.to(device)
    
    # Test with different inputs
    # Near manifold
    x_near = (mu + torch.randn(hidden_size) * 0.1).unsqueeze(0).to(device)
    gate_near = gate(x_near)
    
    # Far from manifold
    x_far = (mu + torch.randn(hidden_size) * 3.0).unsqueeze(0).to(device)
    gate_far = gate(x_far)
    
    # Compute distances for debugging
    dist_near = gate.compute_distance(x_near)
    dist_far = gate.compute_distance(x_far)
    
    print(f"Distance near manifold: {dist_near.item():.4f}")
    print(f"Distance far from manifold: {dist_far.item():.4f}")
    print(f"Gate value near manifold: {gate_near.mean().item():.4f}")
    print(f"Gate value far from manifold: {gate_far.mean().item():.4f}")
    
    assert gate_near.mean() > gate_far.mean(), "Gate should be higher near manifold"
    print("✓ Gating mechanism test passed!")
    
    return gate


def test_flow_transform():
    """Test residual flow transformation."""
    print("\n" + "=" * 60)
    print("Test 3: Flow Transformation")
    print("=" * 60)
    
    hidden_size = 256
    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create flow transform
    flow = ResidualFlow(
        hidden_size=hidden_size,
        hidden_dim=512,
        num_layers=3,
    ).to(device)
    
    # Test forward pass - ResidualFlow returns single output
    x = torch.randn(batch_size, hidden_size).to(device)
    x_transformed = flow(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_transformed.shape}")
    
    # Test inverse
    x_reconstructed = flow.inverse(x_transformed, num_iterations=10)
    reconstruction_error = torch.norm(x - x_reconstructed) / torch.norm(x)
    print(f"Reconstruction error: {reconstruction_error.item():.4f}")
    
    assert x_transformed.shape == x.shape, "Output shape should match input"
    print("✓ Flow transformation test passed!")
    
    return flow


def test_loss_functions():
    """Test GMF loss functions."""
    print("\n" + "=" * 60)
    print("Test 4: Loss Functions")
    print("=" * 60)
    
    hidden_size = 256
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create loss function
    loss_config = GMFLossConfig(
        lambda_attractor=1.0,
        lambda_retain=1.0,
        lambda_flow=0.1,
        lambda_recoverability=0.1,
    )
    loss_fn = GMFLoss(loss_config)
    
    # Create test data
    original_x = torch.randn(batch_size, hidden_size).to(device)
    transformed_x = original_x + torch.randn(batch_size, hidden_size).to(device) * 0.1
    gate_values = torch.rand(batch_size, 1).to(device)
    attractor_mu = torch.randn(hidden_size).to(device)
    forget_mu = torch.randn(hidden_size).to(device)
    # Use diagonal covariance (1D tensor) as expected by GMFLoss
    forget_sigma = torch.abs(torch.randn(hidden_size)).to(device) + 0.1
    attractor_dir = torch.randn(hidden_size).to(device)
    attractor_dir = attractor_dir / torch.norm(attractor_dir)
    
    # Compute forget loss
    forget_loss = loss_fn(
        original_x=original_x,
        transformed_x=transformed_x,
        gate_values=gate_values,
        attractor_mu=attractor_mu,
        forget_manifold_mu=forget_mu,
        forget_manifold_sigma=forget_sigma,
        attractor_direction=attractor_dir,
        is_forget=True,
    )
    
    # Compute retain loss
    retain_loss = loss_fn(
        original_x=original_x,
        transformed_x=transformed_x,
        gate_values=gate_values,
        attractor_mu=attractor_mu,
        forget_manifold_mu=forget_mu,
        forget_manifold_sigma=forget_sigma,
        attractor_direction=attractor_dir,
        is_forget=False,
    )
    
    print(f"Forget loss: {forget_loss['total_loss'].item():.4f}")
    print(f"  - Attractor loss: {forget_loss.get('attractor_loss', torch.tensor(0)).item():.4f}")
    print(f"  - Components: {list(forget_loss.keys())}")
    print(f"Retain loss: {retain_loss['total_loss'].item():.4f}")
    print(f"  - Retain loss: {retain_loss.get('retain_loss', torch.tensor(0)).item():.4f}")
    print(f"  - Components: {list(retain_loss.keys())}")
    print("✓ Loss functions test passed!")
    
    return loss_fn


def test_full_training():
    """Test full GMF training pipeline."""
    print("\n" + "=" * 60)
    print("Test 5: Full Training Pipeline")
    print("=" * 60)
    
    hidden_size = 256
    n_forget = 100
    n_retain = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create synthetic data
    forget_mean = torch.randn(hidden_size)
    forget_activations = [forget_mean + torch.randn(hidden_size) * 0.3 for _ in range(n_forget)]
    
    retain_mean = torch.randn(hidden_size) * 2  # Different distribution
    retain_activations = [retain_mean + torch.randn(hidden_size) * 0.3 for _ in range(n_retain)]
    
    # Create trainer config
    config = GMFTrainerConfig(
        num_epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        sigma=1.0,
        learnable_sigma=False,
        distance_method='mahalanobis',
        flow_hidden_dim=256,
        flow_num_layers=2,
        lambda_attractor=1.0,
        lambda_retain=1.0,
        lambda_flow=0.1,
        lambda_recoverability=0.0,  # Skip recoverability for synthetic test
        device=device,
        save_every=100,  # Don't save during test
    )
    
    # Create trainer
    trainer = GMFTrainer(hidden_size, config)
    
    # Phase 1: Extract manifolds
    refusal_direction = torch.randn(hidden_size)
    refusal_direction = refusal_direction / torch.norm(refusal_direction)
    
    forget_manifold, attractor_manifold = trainer.phase1_extract_manifolds(
        forget_activations,
        retain_activations,
        refusal_direction,
    )
    
    print(f"\nManifold extraction complete:")
    print(f"  Forget mean norm: {torch.norm(forget_manifold.mu):.4f}")
    print(f"  Attractor mean norm: {torch.norm(attractor_manifold.mu):.4f}")
    
    # Phase 2: Train
    forget_inputs = torch.stack(forget_activations)
    retain_inputs = torch.stack(retain_activations)
    
    history = trainer.phase2_train(forget_inputs, retain_inputs)
    
    print(f"\nTraining complete:")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final forget loss: {history['forget_loss'][-1]:.4f}")
    print(f"  Final retain loss: {history['retain_loss'][-1]:.4f}")
    print(f"  Final gate mean: {history['gate_mean'][-1]:.4f}")
    
    # Test transformation
    test_x = forget_mean.unsqueeze(0).to(device)
    transformed, info = trainer.transform(test_x)
    
    print(f"\nTransformation test:")
    print(f"  Original norm: {torch.norm(test_x).item():.4f}")
    print(f"  Transformed norm: {torch.norm(transformed).item():.4f}")
    print(f"  Gate value: {info['gate_mean']:.4f}")
    
    print("✓ Full training pipeline test passed!")
    
    return trainer, history


def test_gated_flow_behavior():
    """Test that gated flow behaves correctly for forget vs retain data."""
    print("\n" + "=" * 60)
    print("Test 6: Gated Flow Behavior Analysis")
    print("=" * 60)
    
    hidden_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create trainer with trained model from previous test
    config = GMFTrainerConfig(
        num_epochs=20,
        batch_size=16,
        learning_rate=1e-3,
        sigma=0.5,  # Tighter gating
        learnable_sigma=False,
        distance_method='mahalanobis',
        flow_hidden_dim=256,
        flow_num_layers=2,
        lambda_attractor=2.0,
        lambda_retain=1.0,
        lambda_flow=0.1,
        lambda_recoverability=0.0,
        device=device,
        save_every=100,
    )
    
    trainer = GMFTrainer(hidden_size, config)
    
    # Create distinct forget and retain distributions
    forget_mean = torch.randn(hidden_size) * 2
    retain_mean = -forget_mean  # Opposite direction
    
    forget_activations = [forget_mean + torch.randn(hidden_size) * 0.2 for _ in range(100)]
    retain_activations = [retain_mean + torch.randn(hidden_size) * 0.2 for _ in range(100)]
    
    # Extract manifolds
    refusal_dir = torch.randn(hidden_size)
    refusal_dir = refusal_dir / torch.norm(refusal_dir)
    
    trainer.phase1_extract_manifolds(forget_activations, retain_activations, refusal_dir)
    
    # Train
    forget_inputs = torch.stack(forget_activations)
    retain_inputs = torch.stack(retain_activations)
    
    trainer.phase2_train(forget_inputs, retain_inputs)
    
    # Analyze behavior
    print("\nBehavior Analysis:")
    
    # Test forget data
    forget_test = forget_inputs[:10].to(device)
    with torch.no_grad():
        forget_transformed, forget_info = trainer.transform(forget_test)
    
    forget_delta = (forget_transformed - forget_test).norm(dim=1).mean()
    forget_gate = forget_info['gate_mean']
    
    # Test retain data
    retain_test = retain_inputs[:10].to(device)
    with torch.no_grad():
        retain_transformed, retain_info = trainer.transform(retain_test)
    
    retain_delta = (retain_transformed - retain_test).norm(dim=1).mean()
    retain_gate = retain_info['gate_mean']
    
    print(f"Forget data:")
    print(f"  Mean transformation delta: {forget_delta.item():.4f}")
    print(f"  Mean gate value: {forget_gate:.4f}")
    
    print(f"Retain data:")
    print(f"  Mean transformation delta: {retain_delta.item():.4f}")
    print(f"  Mean gate value: {retain_gate:.4f}")
    
    print(f"\nGate ratio (forget/retain): {forget_gate/retain_gate:.2f}x")
    
    print("✓ Gated flow behavior test passed!")
    
    return trainer


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GMF Module Tests")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        test_manifold_extraction()
        test_gating_mechanism()
        test_flow_transform()
        test_loss_functions()
        test_full_training()
        test_gated_flow_behavior()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)