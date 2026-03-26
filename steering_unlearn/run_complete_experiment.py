#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Experiment Script for Controlled Region-to-Attractor Transport

This script runs the full experiment pipeline and collects:
- Process metrics (training curves, energy landscapes)
- Final results (ROUGE, ES Score, etc. - from existing evaluator)
- New dynamics metrics (energy reduction, recoverability, attractor distance)

Usage:
    python run_complete_experiment.py
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set environment - MUST be before torch import
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1 which has more free memory

# Add project root to path
sys.path.insert(0, '/home/users/yanzeyu/LUNAR')
sys.path.insert(0, '/home/users/yanzeyu/LUNAR/steering_unlearn')

# Import LUNAR components
from src.model_utils.model_loader import load_model

# Import activation extractor
from preprocessing.activation_extractor import (
    load_or_extract_activations,
    prepare_tofu_instructions,
    load_instructions_from_json,
    ActivationConfig,
    extract_all_layer_activations,
)

# Import dynamics modules
from dynamics import (
    RecoverabilityEstimator,
    RecoverabilityProbeConfig,
    train_recoverability_probes_for_all_layers,
    load_recoverability_probes,
    DualAbstentionAttractor,
    AttractorSetConfig,
    construct_attractor_sets_for_all_layers,
    load_attractor_sets,
    EnergyFunctionConfig,
    TransportController,
    TransportConfig,
    TransportMetrics,
)

# Import existing evaluator
from evaluation.evaluator import (
    evaluate_steering_unlearn,
    save_evaluation_results,
    compute_additional_metrics,
)

# Import hooks for transport
from src.hook_for_unlearn import add_hooks


class ExperimentLogger:
    """Logger for collecting experiment metrics."""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.metrics = {}
        self.training_curves = {}
        self.timestamps = {}
        
    def log(self, stage: str, key: str, value):
        """Log a metric value."""
        if stage not in self.metrics:
            self.metrics[stage] = {}
        self.metrics[stage][key] = value
        print(f"[{stage}] {key}: {value}")
        
    def log_dict(self, stage: str, data: Dict):
        """Log a dictionary of metrics."""
        if stage not in self.metrics:
            self.metrics[stage] = {}
        self.metrics[stage].update(data)
        for k, v in data.items():
            print(f"[{stage}] {k}: {v}")
        
    def log_training_curve(self, name: str, curve: List[float]):
        """Log a training curve."""
        self.training_curves[name] = curve
        
    def save(self):
        """Save all metrics to disk."""
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(self.save_dir, 'experiment_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save training curves
        curves_path = os.path.join(self.save_dir, 'training_curves.json')
        with open(curves_path, 'w') as f:
            json.dump(self.training_curves, f, indent=2)
        
        print(f"\nMetrics saved to {self.save_dir}")


def create_experiment_config():
    """Create experiment configuration."""
    config = {
        # Model
        'model_family': 'llama2-7b-chat',
        'model_path': '/home/users/yanzeyu/LUNAR/models_finetune/tofu_llama2_7b',
        
        # Data
        'data_name': 'tofu_full',
        'forget_edge': ['author_1'],
        'ignorance_reference_path': '/home/users/yanzeyu/LUNAR/dataset/splits/unverified.json',
        'refusal_reference_path': '/home/users/yanzeyu/LUNAR/dataset/splits/harmful.json',
        
        # Layers
        'steering_layers': [18, 19, 21],
        
        # Activation extraction - use smaller subset
        'activation_batch_size': 1,
        'activation_positions': -1,
        'max_forget_samples': 100,  # Limit forget samples
        'max_retain_samples': 200,  # Limit retain samples
        
        # Recoverability probe
        'probe_hidden_dim': 256,
        'probe_num_layers': 3,
        'probe_epochs': 50,  # Increased for better convergence
        'probe_lr': 0.001,
        'probe_batch_size': 32,
        'probe_val_split': 0.2,
        'probe_patience': 15,  # Increased patience
        'probe_weight_decay': 0.01,
        'probe_dropout': 0.1,
        
        # Attractor
        'attractor_subspace_dim': 16,
        'attractor_percentile': 0.80,
        'attractor_min_samples': 10,
        
        # Transport - AGGRESSIVE settings for effective unlearning
        'num_transport_steps': 50,  # More steps for thorough transport
        'transport_step_size': 2.0,  # Larger step size
        'lambda_recoverability': 5.0,  # Balanced recoverability weight
        'lambda_attractor': 5.0,  # INCREASED - attractor is now PRIMARY
        'lambda_drift': 0.0,  # No drift penalty for forget samples
        'recoverability_threshold': 0.1,  # Very low threshold - transport everything
        'control_norm_clip': 10.0,  # Allow even larger control signals
        'primary_attractor': 'refusal',  # USE REFUSAL - more effective for unlearning
        
        # Evaluation (复用现有配置)
        'eval_batch_size': 4,
        'num_eval_samples': 50,
        'generation_max_length': 256,
        'generation_max_new_tokens': 128,
        'if_eval_factual': True,
        'factual_data_path': '/home/users/yanzeyu/LUNAR/dataset/unlearning/factual_data.json',
        'output_es_score': True,
        # Additional attributes needed for LUNAR evaluation
        'eval_generation_max_length': 256,
        'eval_generation_max_new_tokens': 128,
        'do_sample': False,
    }
    
    # Add LUNAR config attributes needed for evaluation
    config['model_name'] = config['model_family']
    config['data_path'] = f"/home/users/yanzeyu/LUNAR/dataset/unlearning/{config['data_name']}.json"
    
    return config


class SimpleConfig:
    """Simple config object for LUNAR evaluation."""
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


def run_experiment():
    """Run the complete experiment pipeline."""
    
    print("="*70)
    print("CONTROLLED REGION-TO-ATTRACTOR TRANSPORT EXPERIMENT")
    print("="*70)
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/home/users/yanzeyu/LUNAR/steering_unlearn/outputs/experiment_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    config = create_experiment_config()
    cfg = SimpleConfig(config)
    logger = ExperimentLogger(save_dir)
    
    # Save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Save directory: {save_dir}")
    
    # ========================================
    # Stage 1: Load Model and Extract Activations
    # ========================================
    print("\n" + "="*70)
    print("STAGE 1: MODEL LOADING AND ACTIVATION EXTRACTION")
    print("="*70)
    
    print(f"\nLoading model from {config['model_path']}...")
    model_base = load_model(config['model_family'], config['model_path'], device)
    print("Model loaded successfully!")
    
    # Log model info
    total_params = sum(p.numel() for p in model_base.model.parameters())
    logger.log('stage1_extraction', 'total_model_params', total_params)
    logger.log('stage1_extraction', 'model_family', config['model_family'])
    
    # Extract activations
    activation_dir = os.path.join(save_dir, 'activations')
    layer_indices = config['steering_layers']
    
    # Load instructions - use include_answers=True for output-level recoverability
    data_path = f"/home/users/yanzeyu/LUNAR/dataset/unlearning/{config['data_name']}.json"
    forget_instructions, forget_answers, retain_instructions = prepare_tofu_instructions(
        data_path=data_path,
        forget_edges=config['forget_edge'],
        include_answers=True,  # KEY: Get forget answers for output-level recoverability
    )
    
    # Limit samples for memory efficiency
    max_forget = config.get('max_forget_samples', len(forget_instructions))
    max_retain = config.get('max_retain_samples', len(retain_instructions))
    forget_instructions = forget_instructions[:max_forget]
    forget_answers = forget_answers[:max_forget]  # Also limit answers
    retain_instructions = retain_instructions[:max_retain]
    
    ignorance_instructions = load_instructions_from_json(config['ignorance_reference_path'])
    refusal_instructions = load_instructions_from_json(config['refusal_reference_path'])
    
    # Log data sizes
    logger.log('stage1_extraction', 'num_forget_samples', len(forget_instructions))
    logger.log('stage1_extraction', 'num_retain_samples', len(retain_instructions))
    logger.log('stage1_extraction', 'num_ignorance_samples', len(ignorance_instructions))
    logger.log('stage1_extraction', 'num_refusal_samples', len(refusal_instructions))
    logger.log('stage1_extraction', 'target_layers', layer_indices)
    
    # Extract activations
    act_config = ActivationConfig(
        batch_size=config['activation_batch_size'],
        positions=config['activation_positions'],
        move_to_cpu=True,
    )
    
    all_activations = extract_all_layer_activations(
        model_base=model_base,
        layer_indices=layer_indices,
        forget_instructions=forget_instructions,
        retain_instructions=retain_instructions,
        ignorance_instructions=ignorance_instructions[:100],
        refusal_instructions=refusal_instructions[:100],
        config=act_config,
        save_dir=activation_dir,
    )
    
    # Log activation shapes
    for layer_idx in layer_indices:
        for dtype in ['forget', 'retain', 'ignorance', 'refusal']:
            shape = list(all_activations[dtype][layer_idx].shape)
            logger.log('stage1_extraction', f'{dtype}_layer{layer_idx}_shape', shape)
    
    # ========================================
    # Stage 2: Train Recoverability Probes
    # ========================================
    print("\n" + "="*70)
    print("STAGE 2: RECOVERABILITY REGION LEARNING")
    print("="*70)
    
    probe_dir = os.path.join(save_dir, 'recoverability_probes')
    
    probe_config = RecoverabilityProbeConfig(
        hidden_dim=config['probe_hidden_dim'],
        num_layers=config['probe_num_layers'],
        epochs=config['probe_epochs'],
        lr=config['probe_lr'],
        batch_size=config['probe_batch_size'],
        val_split=config['probe_val_split'],
        patience=config['probe_patience'],
        weight_decay=config['probe_weight_decay'],
        dropout=config['probe_dropout'],
    )
    
    logger.log_dict('stage2_recoverability', {
        'probe_hidden_dim': probe_config.hidden_dim,
        'probe_num_layers': probe_config.num_layers,
        'probe_epochs': probe_config.epochs,
        'probe_lr': probe_config.lr,
    })
    
    # KEY CHANGE: Use output-level recoverability labels
    # This trains the probe to predict the actual probability that the model
    # would generate the forget answer - the true "recoverability" of knowledge
    print("\n" + "-"*50)
    print("Using OUTPUT-LEVEL recoverability labels")
    print("This measures how likely the model is to generate forget answers")
    print("-"*50)
    
    estimators = train_recoverability_probes_for_all_layers(
        all_activations=all_activations,
        layer_indices=layer_indices,
        config=probe_config,
        device=str(device),
        save_dir=probe_dir,
        verbose=True,
        use_output_level_labels=True,  # KEY: Enable output-level recoverability
        model_base=model_base,  # Required for output-level computation
        forget_questions=forget_instructions,
        forget_answers=forget_answers,
    )
    
    # Probe metrics are already printed during training
    # The training history is returned by the training function but not stored
    # We can evaluate the probes on the data instead
    probe_metrics = {}
    for layer_idx, estimator in estimators.items():
        forget_acts = all_activations['forget'][layer_idx].to(device)
        retain_acts = all_activations['retain'][layer_idx].to(device)
        
        # Get recoverability scores
        forget_scores = estimator.get_recoverability_score(forget_acts)
        retain_scores = estimator.get_recoverability_score(retain_acts)
        
        # Compute accuracy (forget should have high scores, retain low scores)
        forget_acc = (forget_scores > 0.5).float().mean().item()
        retain_acc = (retain_scores < 0.5).float().mean().item()
        
        probe_metrics[f'layer{layer_idx}'] = {
            'forget_high_score_ratio': forget_acc,
            'retain_low_score_ratio': retain_acc,
            'forget_mean_score': forget_scores.mean().item(),
            'retain_mean_score': retain_scores.mean().item(),
        }
    
    logger.log_dict('stage2_recoverability', {'probe_metrics': probe_metrics})
    
    # Compute recoverability scores distribution
    r_scores = {}
    for layer_idx in layer_indices:
        forget_acts = all_activations['forget'][layer_idx].to(device)
        retain_acts = all_activations['retain'][layer_idx].to(device)
        
        forget_scores = estimators[layer_idx].get_recoverability_score(forget_acts)
        retain_scores = estimators[layer_idx].get_recoverability_score(retain_acts)
        
        r_scores[f'layer{layer_idx}'] = {
            'forget_mean': float(forget_scores.mean().item()),
            'forget_std': float(forget_scores.std().item()),
            'retain_mean': float(retain_scores.mean().item()),
            'retain_std': float(retain_scores.std().item()),
            'forget_high_r_ratio': float((forget_scores > 0.5).float().mean().item()),
        }
    
    logger.log_dict('stage2_recoverability', {'recoverability_scores': r_scores})
    
    # ========================================
    # Stage 3: Construct Attractor Sets
    # ========================================
    print("\n" + "="*70)
    print("STAGE 3: ABSTENTION ATTRACTOR CONSTRUCTION")
    print("="*70)
    
    attractor_dir = os.path.join(save_dir, 'attractor_sets')
    
    attractor_config = AttractorSetConfig(
        subspace_dim=config['attractor_subspace_dim'],
        membership_percentile=config['attractor_percentile'],
        min_samples=config['attractor_min_samples'],
        use_pca=True,
    )
    
    logger.log_dict('stage3_attractor', {
        'subspace_dim': attractor_config.subspace_dim,
        'membership_percentile': attractor_config.membership_percentile,
    })
    
    attractors = construct_attractor_sets_for_all_layers(
        all_activations=all_activations,
        layer_indices=layer_indices,
        config=attractor_config,
        device=str(device),
        save_dir=attractor_dir,
        verbose=True,
    )
    
    # Collect attractor metrics
    attractor_metrics = {}
    for layer_idx in layer_indices:
        attractor = attractors[layer_idx]
        ign_attr = attractor.ignorance_attractor
        ref_attr = attractor.refusal_attractor
        
        forget_acts = all_activations['forget'][layer_idx].to(device)
        
        ign_dist = ign_attr.compute_distance_to_attractor(forget_acts)
        ref_dist = ref_attr.compute_distance_to_attractor(forget_acts)
        
        attractor_metrics[f'layer{layer_idx}'] = {
            'ignorance_center_norm': float(ign_attr.center.norm().item()) if ign_attr.center is not None else None,
            'forget_to_ignorance_dist_mean': float(ign_dist.mean().item()),
            'forget_to_ignorance_dist_std': float(ign_dist.std().item()),
            'forget_to_refusal_dist_mean': float(ref_dist.mean().item()),
            'forget_to_refusal_dist_std': float(ref_dist.std().item()),
        }
    
    logger.log_dict('stage3_attractor', {'attractor_metrics': attractor_metrics})
    
    # ========================================
    # Stage 4: Create Transport Controller
    # ========================================
    print("\n" + "="*70)
    print("STAGE 4: TRANSPORT CONTROLLER CREATION")
    print("="*70)
    
    transport_config = TransportConfig(
        num_transport_steps=config['num_transport_steps'],
        step_size=config['transport_step_size'],
        lambda_recoverability=config['lambda_recoverability'],
        lambda_attractor=config['lambda_attractor'],
        lambda_drift=config['lambda_drift'],
        recoverability_threshold=config['recoverability_threshold'],
        control_norm_clip=config['control_norm_clip'],
        primary_attractor=config['primary_attractor'],
        trainable=False,
    )
    
    logger.log_dict('stage4_transport', {
        'num_transport_steps': transport_config.num_transport_steps,
        'step_size': transport_config.step_size,
        'lambda_recoverability': transport_config.lambda_recoverability,
        'lambda_attractor': transport_config.lambda_attractor,
        'lambda_drift': transport_config.lambda_drift,
        'recoverability_threshold': transport_config.recoverability_threshold,
        'primary_attractor': transport_config.primary_attractor,
    })
    
    hidden_dim = all_activations['forget'][layer_indices[0]].shape[1]
    logger.log('stage4_transport', 'hidden_dim', hidden_dim)
    
    controller = TransportController(
        hidden_dim=hidden_dim,
        layer_indices=layer_indices,
        config=transport_config,
        recoverability_estimators=estimators,
        attractor_sets=attractors,
        device=str(device),
    )
    
    # Save controller
    controller.save(os.path.join(save_dir, 'transport_controller'))
    
    # ========================================
    # Stage 5: Run Transport and Collect Dynamics Metrics
    # ========================================
    print("\n" + "="*70)
    print("STAGE 5: TRANSPORT DYNAMICS EVALUATION")
    print("="*70)
    
    transport_metrics = {}
    energy_trajectories = {}
    
    for layer_idx in layer_indices:
        forget_acts = all_activations['forget'][layer_idx][:config['num_eval_samples']].to(device)
        
        # Run multi-step transport
        transported_acts, metrics_list = controller.transport_multi_step(
            activation=forget_acts,
            layer_idx=layer_idx,
            base_activation=forget_acts.clone(),
            num_steps=config['num_transport_steps'],
            verbose=True,
        )
        
        # Collect metrics per step
        step_metrics = []
        energy_trajectory = []
        r_trajectory = []
        dist_trajectory = []
        
        for m in metrics_list:
            step_metrics.append({
                'step': m.step,
                'energy_before': m.energy_before,
                'energy_after': m.energy_after,
                'energy_reduction': m.energy_reduction,
                'recoverability_before': m.recoverability_before,
                'recoverability_after': m.recoverability_after,
                'distance_to_attractor_before': m.distance_to_attractor_before,
                'distance_to_attractor_after': m.distance_to_attractor_after,
                'control_norm': m.control_norm,
            })
            energy_trajectory.append(m.energy_after)
            r_trajectory.append(m.recoverability_after)
            dist_trajectory.append(m.distance_to_attractor_after)
        
        transport_metrics[f'layer{layer_idx}'] = step_metrics
        energy_trajectories[f'layer{layer_idx}_energy'] = energy_trajectory
        energy_trajectories[f'layer{layer_idx}_recoverability'] = r_trajectory
        energy_trajectories[f'layer{layer_idx}_distance'] = dist_trajectory
        
        logger.log_training_curve(f'layer{layer_idx}_energy_trajectory', energy_trajectory)
        logger.log_training_curve(f'layer{layer_idx}_recoverability_trajectory', r_trajectory)
        logger.log_training_curve(f'layer{layer_idx}_distance_trajectory', dist_trajectory)
    
    logger.log_dict('stage5_transport', {'transport_metrics': transport_metrics})
    
    # ========================================
    # Stage 6: Generate with Transport Hooks
    # ========================================
    print("\n" + "="*70)
    print("STAGE 6: GENERATION EVALUATION")
    print("="*70)
    
    # Create transport hooks for generation
    def create_transport_hook(layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            batch_size, seq_len, hidden_dim = activation.shape
            last_idx = seq_len - 1
            last_token_act = activation[:, last_idx, :].clone()
            
            # Enable gradient for transport
            with torch.enable_grad():
                last_token_act = last_token_act.detach().requires_grad_(True)
                # Apply transport
                transported, _ = controller.transport_multi_step(
                    activation=last_token_act,
                    layer_idx=layer_idx,
                    num_steps=config['num_transport_steps'],
                )
            
            activation[:, last_idx, :] = transported.detach()
            
            if isinstance(output, tuple):
                return (activation, *output[1:])
            else:
                return activation
        return hook_fn
    
    # Register hooks
    fwd_hooks = []
    for layer_idx in layer_indices:
        layer = model_base.model.model.layers[layer_idx]
        hook_fn = create_transport_hook(layer_idx)
        fwd_hooks.append((layer, hook_fn))
    
    # Sample generation test
    print("\nSample generation with transport:")
    sample_questions = forget_instructions[:3]
    
    from src.eval_util import custom_evaluate
    
    # Evaluate on forget set (using LUNAR's evaluation)
    print("\nEvaluating on forget set...")
    eval_logs_forget = custom_evaluate(
        cfg=cfg,
        data_path=config['data_path'],
        tokenizer=model_base.tokenizer,
        model=model_base,
        eval_target="forget_edge",
        fwd_pre_hooks=[],
        fwd_hooks=fwd_hooks,
        output_es_score=config['output_es_score'],
    )
    
    # Evaluate on retain set
    print("\nEvaluating on retain set...")
    eval_logs_retain = custom_evaluate(
        cfg=cfg,
        data_path=config['data_path'],
        tokenizer=model_base.tokenizer,
        model=model_base,
        eval_target="retained_edge",
        fwd_pre_hooks=[],
        fwd_hooks=fwd_hooks,
        output_es_score=False,
    )
    
    # Compile results
    evaluation_results = {
        "forget": eval_logs_forget,
        "retain": eval_logs_retain,
    }
    
    # Log evaluation results
    logger.log_dict('stage6_evaluation', {
        'forget_rouge1_recall': eval_logs_forget.get('rouge1_recall'),
        'forget_rougeL_recall': eval_logs_forget.get('rougeL_recall'),
        'retain_rouge1_recall': eval_logs_retain.get('rouge1_recall'),
        'retain_rougeL_recall': eval_logs_retain.get('rougeL_recall'),
    })
    
    if 'es_score' in eval_logs_forget:
        logger.log('stage6_evaluation', 'forget_es_score', eval_logs_forget['es_score'])
    
    # ========================================
    # Final Analysis
    # ========================================
    print("\n" + "="*70)
    print("FINAL ANALYSIS")
    print("="*70)
    
    # Compute summary statistics
    summary = {
        'experiment_timestamp': timestamp,
        'num_layers': len(layer_indices),
        'total_transport_steps': config['num_transport_steps'],
    }
    
    # Average dynamics metrics across layers
    energy_reductions = []
    r_reductions = []
    dist_reductions = []
    
    for layer_idx in layer_indices:
        if f'layer{layer_idx}' in transport_metrics:
            metrics = transport_metrics[f'layer{layer_idx}']
            if metrics:
                first_step = metrics[0]
                last_step = metrics[-1]
                
                energy_red = first_step['energy_before'] - last_step['energy_after']
                r_red = first_step['recoverability_before'] - last_step['recoverability_after']
                dist_red = first_step['distance_to_attractor_before'] - last_step['distance_to_attractor_after']
                
                energy_reductions.append(energy_red)
                r_reductions.append(r_red)
                dist_reductions.append(dist_red)
    
    summary['avg_energy_reduction'] = float(np.mean(energy_reductions)) if energy_reductions else 0.0
    summary['avg_recoverability_reduction'] = float(np.mean(r_reductions)) if r_reductions else 0.0
    summary['avg_distance_reduction'] = float(np.mean(dist_reductions)) if dist_reductions else 0.0
    
    # Evaluation metrics
    summary['forget_rouge1'] = eval_logs_forget.get('rouge1_recall')
    summary['forget_rougeL'] = eval_logs_forget.get('rougeL_recall')
    summary['retain_rouge1'] = eval_logs_retain.get('rouge1_recall')
    summary['retain_rougeL'] = eval_logs_retain.get('rougeL_recall')
    
    if 'es_score' in eval_logs_forget:
        summary['forget_es_score'] = eval_logs_forget['es_score']
    
    logger.log_dict('final_summary', summary)
    
    # Save all metrics
    logger.save()
    
    # ========================================
    # Print Final Report
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE - FINAL REPORT")
    print("="*70)
    
    print(f"\n📊 Experiment Directory: {save_dir}")
    
    print("\n📈 Dynamics Metrics:")
    print(f"  • Average Energy Reduction: {summary['avg_energy_reduction']:.4f}")
    print(f"  • Average Recoverability Reduction: {summary['avg_recoverability_reduction']:.4f}")
    print(f"  • Average Distance to Attractor Reduction: {summary['avg_distance_reduction']:.4f}")
    
    print("\n📊 Evaluation Metrics (ROUGE):")
    print(f"  • Forget ROUGE-1 Recall: {summary['forget_rouge1']:.4f}" if summary['forget_rouge1'] else "  • Forget ROUGE-1 Recall: N/A")
    print(f"  • Forget ROUGE-L Recall: {summary['forget_rougeL']:.4f}" if summary['forget_rougeL'] else "  • Forget ROUGE-L Recall: N/A")
    print(f"  • Retain ROUGE-1 Recall: {summary['retain_rouge1']:.4f}" if summary['retain_rouge1'] else "  • Retain ROUGE-1 Recall: N/A")
    print(f"  • Retain ROUGE-L Recall: {summary['retain_rougeL']:.4f}" if summary['retain_rougeL'] else "  • Retain ROUGE-L Recall: N/A")
    
    if 'forget_es_score' in summary:
        print(f"  • Forget ES Score: {summary['forget_es_score']:.4f}")
    
    print("\n📁 Saved Files:")
    print(f"  • config.json - Experiment configuration")
    print(f"  • experiment_metrics.json - All collected metrics")
    print(f"  • training_curves.json - Training trajectory data")
    print(f"  • activations/ - Extracted activation tensors")
    print(f"  • recoverability_probes/ - Trained probe models")
    print(f"  • attractor_sets/ - Constructed attractor models")
    print(f"  • transport_controller/ - Transport controller")
    
    # Create markdown summary
    forget_r1 = summary.get('forget_rouge1')
    retain_r1 = summary.get('retain_rouge1')
    forget_rL = summary.get('forget_rougeL')
    retain_rL = summary.get('retain_rougeL')
    es_score = summary.get('forget_es_score', 'N/A')
    
    markdown_report = f"""
# Experiment Report: Controlled Region-to-Attractor Transport

## Configuration
- Model: {config['model_family']}
- Target Layers: {layer_indices}
- Forget Edge: {config['forget_edge']}

## Dynamics Metrics

| Metric | Value |
|--------|-------|
| Avg Energy Reduction | {summary['avg_energy_reduction']:.4f} |
| Avg Recoverability Reduction | {summary['avg_recoverability_reduction']:.4f} |
| Avg Distance Reduction | {summary['avg_distance_reduction']:.4f} |

## Evaluation Metrics

| Metric | Forget | Retain |
|--------|--------|--------|
| ROUGE-1 Recall | {forget_r1:.4f if forget_r1 else 'N/A'} | {retain_r1:.4f if retain_r1 else 'N/A'} |
| ROUGE-L Recall | {forget_rL:.4f if forget_rL else 'N/A'} | {retain_rL:.4f if retain_rL else 'N/A'} |
| ES Score | {es_score} | - |

## Interpretation

1. **Energy Reduction**: Lower energy indicates successful transport from recoverability region to attractor.
2. **Recoverability Reduction**: Lower recoverability means the model is less likely to recover forgotten knowledge.
3. **Distance Reduction**: Lower distance to attractor indicates the activation moved towards abstention behavior.
4. **ROUGE Metrics**: 
   - Lower forget ROUGE = better forgetting
   - Higher retain ROUGE = better knowledge preservation

---
Generated: {timestamp}
"""
    
    with open(os.path.join(save_dir, 'report.md'), 'w') as f:
        f.write(markdown_report)
    
    print(f"\n📄 Markdown report saved to {os.path.join(save_dir, 'report.md')}")
    
    return logger.metrics, summary


if __name__ == "__main__":
    metrics, summary = run_experiment()
    print("\n✅ Experiment completed successfully!")