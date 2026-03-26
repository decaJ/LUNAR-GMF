#!/usr/bin/env python3
"""
Test script for Gated Manifold Flow (GMF) modules.
Tests the basic functionality of each component.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gmf.manifold import ForgetManifold, AttractorManifold, ManifoldExtractor
from gmf.gating import DistanceBasedGate, GatedFlow
from gmf.flow_transform import ResidualFlow, AffineFlow
from gmf.losses import GMFLoss, GMFLossConfig
from gmf.trainer import GMFTrainer, GMFTrainerConfig, GMFModule


def test_forget_manifold():
    """Test ForgetManifold class."""
    print("\n" + "=" * 60)
    print("Testing ForgetManifold")
    print("=" * 60)
    
    hidden_size = 4096
    batch_size = 8
    
    # Create random manifold
    mu = torch.randn(hidden_size)
    Sigma = torch.ones(hidden_size) * 0.1  # Diagonal covariance
    
    manifold = ForgetManifold(mu=mu, Sigma=Sigma)
    
    # Test distance computation
    x = torch.randn(batch_size, hidden_size)
    
    euclidean_dist = manifold.distance(x, method='euclidean')
    mahalanobis_dist = manifold.distance(x, method='mahalanobis')
    
    print(f"Input shape: {x.shape}")
    print(f"Euclidean distances shape: {euclidean_dist.shape}")
    print(f"Mahalanobis distances shape: {mahalanobis_dist.shape}")
    print(f"Mean Euclidean distance: {euclidean_dist.mean():.4f}")
    print(f"Mean Mahalanobis distance: {mahalanobis_dist.mean():.4f}")
    
    assert euclidean_dist.shape == (batch_size,), "Wrong euclidean distance shape"
    assert mahalanobis_dist.shape == (batch_size,), "Wrong mahalanobis distance shape"
    
    print("✓ ForgetManifold test passed!")
    return True


def test_attractor_manifold():
    """Test AttractorManifold class."""
    print("\n" + "=" * 60)
    print("Testing AttractorManifold")
    print("=" * 60)
    
    hidden_size = 4096
    batch_size = 8
    
    # Create random attractor
    mu = torch.randn(hidden_size)
    direction = torch.randn(hidden_size)
    direction = direction / torch.norm(direction)
    
    attractor = AttractorManifold(mu=mu, direction=direction, scale=1.0)
    
    # Test projection
    x = torch.randn(batch_size, hidden_size)
    projected = attractor.project(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Projected shape: {projected.shape}")
    
    assert projected.shape == x.shape, "Wrong projected shape"
    
    print("✓ AttractorManifold test passed!")
    return True


def test_distance_based_gate():
    """Test DistanceBasedGate class."""
    print("\n" + "=" * 60)
    print("Testing DistanceBasedGate")
    print("=" * 60)
    
    hidden_size = 4096
    batch_size = 8
    
    # Create gate
    gate = DistanceBasedGate(
        hidden_size=hidden_size,
        sigma=1.0,
        learnable_sigma=False,
        distance_method='mahalanobis',
    )
    
    # Create manifold and set it
    mu = torch.randn(hidden_size)
    Sigma = torch.ones(hidden_size) * 0.1
    manifold = ForgetManifold(mu=mu, Sigma=Sigma)
    gate.set_manifold(manifold)
    
    # Test gate computation
    x = torch.randn(batch_size, hidden_size)
    gate_values = gate(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Gate values shape: {gate_values.shape}")
    print(f"Gate values range: [{gate_values.min():.4f}, {gate_values.max():.4f}]")
    print(f"Gate values mean: {gate_values.mean():.4f}")
    
    assert gate_values.shape == (batch_size, 1), "Wrong gate values shape"
    assert (gate_values >= 0).all() and (gate_values <= 1).all(), "Gate values out of range"
    
    print("✓ DistanceBasedGate test passed!")
    return True


def test_residual_flow():
    """Test ResidualFlow class."""
    print("\n" + "=" * 60)
    print("Testing ResidualFlow")
    print("=" * 60)
    
    hidden_size = 4096
    batch_size = 8
    
    # Create flow
    flow = ResidualFlow(
        hidden_size=hidden_size,
        hidden_dim=512,
        num_layers=3,
    )
    
    # Test forward pass
    x = torch.randn(batch_size, hidden_size)
    y = flow(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Input norm mean: {torch.norm(x, dim=-1).mean():.4f}")
    print(f"Output norm mean: {torch.norm(y, dim=-1).mean():.4f}")
    
    assert y.shape == x.shape, "Wrong output shape"
    
    # Test inverse
    x_reconstructed = flow.inverse(y)
    print(f"Reconstruction error: {torch.norm(x - x_reconstructed).item():.6f}")
    
    print("✓ ResidualFlow test passed!")
    return True


def test_gated_flow():
    """Test GatedFlow class."""
    print("\n" + "=" * 60)
    print("Testing GatedFlow")
    print("=" * 60)
    
    hidden_size = 4096
    batch_size = 8
    
    # Create components
    flow_transform = ResidualFlow(hidden_size=hidden_size, hidden_dim=512)
    gate = DistanceBasedGate(hidden_size=hidden_size, sigma=1.0)
    
    # Set manifold
    mu = torch.randn(hidden_size)
    Sigma = torch.ones(hidden_size) * 0.1
    manifold = ForgetManifold(mu=mu, Sigma=Sigma)
    gate.set_manifold(manifold)
    
    # Create gated flow
    gated_flow = GatedFlow(flow_transform=flow_transform, gate=gate)
    
    # Test forward pass
    x = torch.randn(batch_size, hidden_size)
    y, info = gated_flow(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Gate mean: {info['gate_mean']:.4f}")
    print(f"Gate std: {info['gate_std']:.4f}")
    
    assert y.shape == x.shape, "Wrong output shape"
    assert 'gate_values' in info, "Missing gate_values in info"
    
    print("✓ GatedFlow test passed!")
    return True


def test_gmf_loss():
    """Test GMFLoss class."""
    print("\n" + "=" * 60)
    print("Testing GMFLoss")
    print("=" * 60)
    
    hidden_size = 4096
    batch_size = 8
    
    # Create loss function
    config = GMFLossConfig(
        lambda_attractor=1.0,
        lambda_retain=1.0,
        lambda_flow=0.1,
        lambda_recoverability=0.1,
    )
    loss_fn = GMFLoss(config)
    
    # Create test data
    original_x = torch.randn(batch_size, hidden_size)
    transformed_x = original_x + torch.randn(batch_size, hidden_size) * 0.1
    gate_values = torch.rand(batch_size, 1)
    attractor_mu = torch.randn(hidden_size)
    forget_manifold_mu = torch.randn(hidden_size)
    forget_manifold_sigma = torch.ones(hidden_size) * 0.1
    
    # Test forget loss
    losses = loss_fn(
        original_x=original_x,
        transformed_x=transformed_x,
        gate_values=gate_values,
        attractor_mu=attractor_mu,
        forget_manifold_mu=forget_manifold_mu,
        forget_manifold_sigma=forget_manifold_sigma,
        is_forget=True,
    )
    
    print("Forget losses:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
    
    # Test retain loss
    losses = loss_fn(
        original_x=original_x,
        transformed_x=transformed_x,
        gate_values=gate_values,
        attractor_mu=attractor_mu,
        forget_manifold_mu=forget_manifold_mu,
        forget_manifold_sigma=forget_manifold_sigma,
        is_forget=False,
    )
    
    print("\nRetain losses:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
    
    assert 'total_loss' in losses, "Missing total_loss"
    
    print("✓ GMFLoss test passed!")
    return True


def test_gmf_module():
    """Test GMFModule class."""
    print("\n" + "=" * 60)
    print("Testing GMFModule")
    print("=" * 60)
    
    hidden_size = 4096
    batch_size = 8
    
    # Create config
    config = GMFTrainerConfig(
        num_epochs=5,
        batch_size=8,
        learning_rate=1e-4,
        sigma=1.0,
        flow_hidden_dim=512,
        flow_num_layers=3,
        device='cpu',  # Use CPU for testing
    )
    
    # Create GMF module
    gmf_module = GMFModule(hidden_size, config)
    
    # Set manifold
    mu = torch.randn(hidden_size)
    Sigma = torch.ones(hidden_size) * 0.1
    manifold = ForgetManifold(mu=mu, Sigma=Sigma)
    gmf_module.set_manifold(manifold)
    
    # Test forward pass
    x = torch.randn(batch_size, hidden_size)
    y, info = gmf_module(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Gate mean: {info['gate_mean']:.4f}")
    
    assert y.shape == x.shape, "Wrong output shape"
    
    print("✓ GMFModule test passed!")
    return True


def test_training_step():
    """Test a single training step."""
    print("\n" + "=" * 60)
    print("Testing Training Step")
    print("=" * 60)
    
    hidden_size = 256  # Smaller for testing
    num_forget = 16
    num_retain = 16
    
    # Create config
    config = GMFTrainerConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        sigma=1.0,
        flow_hidden_dim=64,
        flow_num_layers=2,
        device='cpu',
    )
    
    # Create trainer
    trainer = GMFTrainer(hidden_size, config)
    
    # Create dummy data
    forget_inputs = torch.randn(num_forget, hidden_size)
    retain_inputs = torch.randn(num_retain, hidden_size)
    
    # Set manifolds
    mu = forget_inputs.mean(dim=0)
    Sigma = forget_inputs.var(dim=0) + 1e-6
    forget_manifold = ForgetManifold(mu=mu, Sigma=Sigma)
    attractor_manifold = AttractorManifold(
        mu=mu + torch.randn(hidden_size) * 0.1,
        direction=torch.randn(hidden_size),
    )
    
    trainer.forget_manifold = forget_manifold
    trainer.attractor_manifold = attractor_manifold
    trainer.gmf_module.set_manifold(forget_manifold)
    
    # Run one training step
    print("Running single training step...")
    
    # Create optimizer
    trainer.optimizer = torch.optim.AdamW(
        trainer.gmf_module.parameters(),
        lr=config.learning_rate,
    )
    
    # Single training iteration
    trainer.gmf_module.train()
    
    forget_x = forget_inputs[:config.batch_size]
    retain_x = retain_inputs[:config.batch_size]
    
    # Forward pass
    forget_transformed, forget_info = trainer.gmf_module(forget_x)
    retain_transformed, retain_info = trainer.gmf_module(retain_x)
    
    # Compute loss
    forget_loss = trainer.loss_fn(
        original_x=forget_x,
        transformed_x=forget_transformed,
        gate_values=forget_info['gate_values'],
        attractor_mu=trainer.attractor_manifold.mu,
        forget_manifold_mu=trainer.forget_manifold.mu,
        forget_manifold_sigma=trainer.forget_manifold.Sigma,
        is_forget=True,
    )
    
    retain_loss = trainer.loss_fn(
        original_x=retain_x,
        transformed_x=retain_transformed,
        gate_values=retain_info['gate_values'],
        attractor_mu=trainer.attractor_manifold.mu,
        forget_manifold_mu=trainer.forget_manifold.mu,
        forget_manifold_sigma=trainer.forget_manifold.Sigma,
        is_forget=False,
    )
    
    total_loss = forget_loss['total_loss'] + retain_loss['total_loss']
    
    # Backward pass
    trainer.optimizer.zero_grad()
    total_loss.backward()
    trainer.optimizer.step()
    
    print(f"Forget loss: {forget_loss['total_loss'].item():.4f}")
    print(f"Retain loss: {retain_loss['total_loss'].item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    print("✓ Training step test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" GATED MANIFOLD FLOW - UNIT TESTS")
    print("=" * 80)
    
    tests = [
        ("ForgetManifold", test_forget_manifold),
        ("AttractorManifold", test_attractor_manifold),
        ("DistanceBasedGate", test_distance_based_gate),
        ("ResidualFlow", test_residual_flow),
        ("GatedFlow", test_gated_flow),
        ("GMFLoss", test_gmf_loss),
        ("GMFModule", test_gmf_module),
        ("Training Step", test_training_step),
    ]
    
    results = {}
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = "PASSED" if result else "FAILED"
        except Exception as e:
            print(f"✗ {name} test failed with error: {e}")
            results[name] = f"ERROR: {str(e)[:50]}"
    
    # Print summary
    print("\n" + "=" * 80)
    print(" TEST SUMMARY")
    print("=" * 80)
    
    for name, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        print(f"  {status} {name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(tests)
    
    print("=" * 80)
    print(f" Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)