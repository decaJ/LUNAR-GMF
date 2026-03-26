# Copyright (c) Steering Unlearn Project
"""
Main Entry Point for Controlled Region-to-Attractor Transport Unlearning

This script implements the complete dynamics-based unlearning pipeline:
1. Extract activations from target layers
2. Train recoverability probes (learn R_f^{(l)})
3. Construct abstention attractor sets (learn A^{(l)})
4. Create transport controller with energy function
5. Evaluate unlearning effectiveness

Method Name: Controlled Region-to-Attractor Transport for LLM Unlearning

Key concepts:
- Recoverability Region: R_f^{(l)} = { a : r^{(l)}(a) >= tau_f }
- Abstention Attractor: A^{(l)} = { a : s^{(l)}(a) >= tau_a }
- Energy Function: E^{(l)}(a) = φ_r(r) + λ_a * φ_a(s) + λ_d * φ_d(a, a_base)
- Control Law: u_l(a) = -η_l * ∇_a E^{(l)}(a)
- Finite-step Transport: a^+ = a + u(a)
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra

# Set HuggingFace mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LUNAR components
from src.model_utils.model_loader import load_model

# Import activation extractor
from steering_unlearn.preprocessing.activation_extractor import (
    load_or_extract_activations,
    prepare_tofu_instructions,
    load_instructions_from_json,
)

# Import dynamics-based unlearning modules
from steering_unlearn.dynamics import (
    # Recoverability
    RecoverabilityEstimator,
    RecoverabilityProbeConfig,
    train_recoverability_probes_for_all_layers,
    load_recoverability_probes,
    
    # Attractor
    DualAbstentionAttractor,
    AttractorSetConfig,
    construct_attractor_sets_for_all_layers,
    load_attractor_sets,
    
    # Energy Function
    EnergyFunctionConfig,
    
    # Transport Controller
    TransportController,
    TransportConfig,
    create_transport_controller,
    apply_transport_to_model,
)

# Import evaluator
from steering_unlearn.evaluation.evaluator import (
    evaluate_steering_unlearn,
    save_evaluation_results,
)


def run_activation_extraction(cfg, model_base, save_dir: str, force_extract: bool = False):
    """
    Extract activations from target layers.
    
    Returns:
        Dictionary with activations organized by data type and layer
    """
    print("\n" + "="*70)
    print("STAGE 1: ACTIVATION EXTRACTION")
    print("="*70)
    
    activation_dir = os.path.join(save_dir, "activations")
    layer_indices = cfg.steering_layers
    
    all_activations = load_or_extract_activations(
        model_base=model_base,
        cfg=cfg,
        layer_indices=layer_indices,
        save_dir=activation_dir,
        force_extract=force_extract,
    )
    
    print(f"\nExtracted activations for {len(layer_indices)} layers")
    print(f"Data types: {list(all_activations.keys())}")
    
    return all_activations


def run_recoverability_training(
    all_activations,
    layer_indices: List[int],
    cfg,
    save_dir: str,
    device: str = "cuda",
):
    """
    Train recoverability probes for all target layers.
    
    This learns the Recoverability Region: R_f^{(l)} = { a : r^{(l)}(a) >= tau_f }
    
    Returns:
        Dictionary mapping layer index to RecoverabilityEstimator
    """
    print("\n" + "="*70)
    print("STAGE 2: RECOVERABILITY REGION LEARNING")
    print("="*70)
    
    probe_dir = os.path.join(save_dir, "recoverability_probes")
    
    # Check if already trained
    if os.path.exists(probe_dir) and not cfg.get('force_retrain', False):
        print("Loading pre-trained recoverability probes...")
        return load_recoverability_probes(probe_dir, layer_indices, device)
    
    # Configure probe training
    probe_config = RecoverabilityProbeConfig(
        hidden_dim=cfg.get('probe_hidden_dim', 256),
        num_layers=cfg.get('probe_num_layers', 3),
        epochs=cfg.get('probe_epochs', 50),
        lr=cfg.get('probe_lr', 0.001),
        batch_size=cfg.get('probe_batch_size', 32),
        val_split=cfg.get('probe_val_split', 0.2),
        patience=cfg.get('probe_patience', 10),
        weight_decay=cfg.get('probe_weight_decay', 0.01),
        dropout=cfg.get('probe_dropout', 0.1),
    )
    
    print(f"\nRecoverability Probe Config:")
    print(f"  Hidden dim: {probe_config.hidden_dim}")
    print(f"  Num layers: {probe_config.num_layers}")
    print(f"  Epochs: {probe_config.epochs}")
    print(f"  Learning rate: {probe_config.lr}")
    
    # Train probes
    estimators = train_recoverability_probes_for_all_layers(
        all_activations=all_activations,
        layer_indices=layer_indices,
        config=probe_config,
        device=device,
        save_dir=probe_dir,
        verbose=True,
    )
    
    return estimators


def run_attractor_construction(
    all_activations,
    layer_indices: List[int],
    cfg,
    save_dir: str,
    device: str = "cuda",
):
    """
    Construct abstention attractor sets for all target layers.
    
    This learns the Attractor Set: A^{(l)} = { a : s^{(l)}(a) >= tau_a }
    
    Returns:
        Dictionary mapping layer index to DualAbstentionAttractor
    """
    print("\n" + "="*70)
    print("STAGE 3: ABSTENTION ATTRACTOR CONSTRUCTION")
    print("="*70)
    
    attractor_dir = os.path.join(save_dir, "attractor_sets")
    
    # Check if already constructed
    if os.path.exists(attractor_dir) and not cfg.get('force_retrain', False):
        print("Loading pre-constructed attractor sets...")
        hidden_dim = all_activations['ignorance'][layer_indices[0]].shape[1]
        attractor_config = AttractorSetConfig(
            subspace_dim=cfg.get('attractor_subspace_dim', 16),
            membership_percentile=cfg.get('attractor_percentile', 0.80),
        )
        return load_attractor_sets(attractor_dir, layer_indices, hidden_dim, attractor_config, device)
    
    # Configure attractor construction
    attractor_config = AttractorSetConfig(
        subspace_dim=cfg.get('attractor_subspace_dim', 16),
        membership_percentile=cfg.get('attractor_percentile', 0.80),
        min_samples=cfg.get('attractor_min_samples', 10),
        use_pca=True,
    )
    
    print(f"\nAttractor Set Config:")
    print(f"  Subspace dim: {attractor_config.subspace_dim}")
    print(f"  Membership percentile: {attractor_config.membership_percentile}")
    
    # Construct attractors
    attractors = construct_attractor_sets_for_all_layers(
        all_activations=all_activations,
        layer_indices=layer_indices,
        config=attractor_config,
        device=device,
        save_dir=attractor_dir,
        verbose=True,
    )
    
    return attractors


def run_transport_controller_creation(
    all_activations,
    recoverability_estimators,
    attractor_sets,
    layer_indices: List[int],
    cfg,
    save_dir: str,
    device: str = "cuda",
):
    """
    Create the transport controller with energy function.
    
    The controller implements:
    - Energy Function: E^{(l)}(a) = φ_r(r) + λ_a * φ_a(s) + λ_d * φ_d(a, a_base)
    - Control Law: u_l(a) = -η_l * ∇_a E^{(l)}(a)
    
    Returns:
        TransportController instance
    """
    print("\n" + "="*70)
    print("STAGE 4: TRANSPORT CONTROLLER CREATION")
    print("="*70)
    
    # Configure transport
    transport_config = TransportConfig(
        num_transport_steps=cfg.get('num_transport_steps', 5),
        step_size=cfg.get('transport_step_size', 0.1),
        lambda_recoverability=cfg.get('lambda_recoverability', 1.0),
        lambda_attractor=cfg.get('lambda_attractor', 0.5),
        lambda_drift=cfg.get('lambda_drift', 2.0),
        recoverability_threshold=cfg.get('recoverability_threshold', 0.5),
        control_norm_clip=cfg.get('control_norm_clip', 1.0),
        primary_attractor=cfg.get('primary_attractor', 'ignorance'),
        trainable=cfg.get('trainable_controller', False),
    )
    
    # Configure recoverability probe
    recoverability_config = RecoverabilityProbeConfig(
        hidden_dim=cfg.get('probe_hidden_dim', 256),
        num_layers=cfg.get('probe_num_layers', 3),
        epochs=cfg.get('probe_epochs', 50),
        lr=cfg.get('probe_lr', 0.001),
    )
    
    # Configure attractor
    attractor_config = AttractorSetConfig(
        subspace_dim=cfg.get('attractor_subspace_dim', 16),
        membership_percentile=cfg.get('attractor_percentile', 0.80),
    )
    
    print(f"\nTransport Config:")
    print(f"  Num transport steps: {transport_config.num_transport_steps}")
    print(f"  Step size: {transport_config.step_size}")
    print(f"  λ_recoverability: {transport_config.lambda_recoverability}")
    print(f"  λ_attractor: {transport_config.lambda_attractor}")
    print(f"  λ_drift: {transport_config.lambda_drift}")
    print(f"  Recoverability threshold: {transport_config.recoverability_threshold}")
    print(f"  Primary attractor: {transport_config.primary_attractor}")
    
    # Create controller
    hidden_dim = all_activations['forget'][layer_indices[0]].shape[1]
    
    controller = TransportController(
        hidden_dim=hidden_dim,
        layer_indices=layer_indices,
        config=transport_config,
        recoverability_estimators=recoverability_estimators,
        attractor_sets=attractor_sets,
        device=device,
    )
    
    # Save controller
    controller_save_dir = os.path.join(save_dir, "transport_controller")
    controller.save(controller_save_dir)
    
    return controller


def run_evaluation(
    cfg,
    model_base,
    controller: TransportController,
    save_dir: str,
):
    """
    Evaluate the transport-based unlearning.
    """
    print("\n" + "="*70)
    print("STAGE 5: EVALUATION")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_indices = cfg.steering_layers
    eval_dir = os.path.join(save_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Get data path
    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")
    
    # Load forget and retain questions
    forget_questions, retain_questions = prepare_tofu_instructions(
        data_path=data_path,
        forget_edges=cfg.forget_edge,
    )
    
    # Apply transport to forget questions
    print("\nApplying transport to forget questions...")
    forget_responses = apply_transport_to_model(
        model_base=model_base,
        controller=controller,
        instructions=forget_questions[:10],  # Evaluate on subset
        layer_indices=layer_indices,
        batch_size=cfg.get('eval_batch_size', 4),
    )
    
    print("\nForget Response Examples:")
    for i, (q, r) in enumerate(zip(forget_questions[:3], forget_responses[:3])):
        print(f"\nQ: {q[:100]}...")
        print(f"A: {r[:200]}...")
    
    # Save results
    results = {
        'forget_responses': forget_responses,
        'config': OmegaConf.to_container(cfg),
    }
    
    results_path = os.path.join(eval_dir, "transport_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'forget_responses': forget_responses,
        }, f, indent=2)
    
    print(f"\nEvaluation results saved to {results_path}")
    
    return results


@hydra.main(version_base=None, config_path="config", config_name="dynamics_config")
def main(cfg: DictConfig):
    """Main entry point for dynamics-based unlearning."""
    
    print("="*70)
    print("CONTROLLED REGION-TO-ATTRACTOR TRANSPORT FOR LLM UNLEARNING")
    print("="*70)
    print(f"\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(cfg.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # Save configuration
    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Configuration saved to {config_path}")
    
    # -----------------------------
    # Load Model
    # -----------------------------
    print(f"\nLoading model from {cfg.model_path}...")
    model_base = load_model(cfg.model_family, cfg.model_path, device)
    print("Model loaded successfully!")
    
    layer_indices = cfg.steering_layers
    
    # -----------------------------
    # Stage 1: Extract Activations
    # -----------------------------
    all_activations = run_activation_extraction(
        cfg=cfg,
        model_base=model_base,
        save_dir=save_dir,
        force_extract=False,
    )
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # -----------------------------
    # Stage 2: Train Recoverability Probes
    # -----------------------------
    recoverability_estimators = run_recoverability_training(
        all_activations=all_activations,
        layer_indices=layer_indices,
        cfg=cfg,
        save_dir=save_dir,
        device=str(device),
    )
    
    # -----------------------------
    # Stage 3: Construct Attractor Sets
    # -----------------------------
    attractor_sets = run_attractor_construction(
        all_activations=all_activations,
        layer_indices=layer_indices,
        cfg=cfg,
        save_dir=save_dir,
        device=str(device),
    )
    
    # -----------------------------
    # Stage 4: Create Transport Controller
    # -----------------------------
    controller = run_transport_controller_creation(
        all_activations=all_activations,
        recoverability_estimators=recoverability_estimators,
        attractor_sets=attractor_sets,
        layer_indices=layer_indices,
        cfg=cfg,
        save_dir=save_dir,
        device=str(device),
    )
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # -----------------------------
    # Stage 5: Evaluation
    # -----------------------------
    results = run_evaluation(
        cfg=cfg,
        model_base=model_base,
        controller=controller,
        save_dir=save_dir,
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {save_dir}")
    
    return results


if __name__ == "__main__":
    main()