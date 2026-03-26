# Copyright (c) Steering Unlearn Project
"""
Evaluator for Steering Unlearn Method

This module handles the evaluation phase, reusing LUNAR's evaluation code.
It outputs metrics compatible with LUNAR for direct comparison.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

# Import from LUNAR
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.eval_util import (
    custom_evaluate,
    get_dataloader,
    get_all_evals,
)
from src.hook_for_unlearn import add_hooks, get_activations_fwd_hook
from src.utils.hook_utils import add_hooks as lunar_add_hooks


class SteeringHookManager:
    """
    Manages hooks for applying steering during inference.
    
    This class registers forward hooks on target layers to apply
    steering during model forward passes.
    """
    
    def __init__(
        self,
        steering_modules: Dict[int, nn.Module],
        risk_probes: Dict[int, nn.Module],
        layer_indices: List[int],
        device: str = "cuda",
    ):
        """
        Args:
            steering_modules: Trained steering modules per layer
            risk_probes: Trained risk probes per layer
            layer_indices: Target layer indices
            device: Device to use
        """
        self.steering_modules = steering_modules
        self.risk_probes = risk_probes
        self.layer_indices = layer_indices
        self.device = device
        
        # Set all modules to eval mode
        for module in self.steering_modules.values():
            module.eval()
        for probe in self.risk_probes.values():
            probe.eval()
    
    def create_steering_hook(self, layer_idx: int, debug: bool = False):
        """
        Create a forward hook that applies steering to activations.
        
        CRITICAL: This hook applies steering ONLY to the last valid token position.
        - During prefill: last token of the input prompt
        - During decoding: the single new token being generated
        
        The key insight is that in autoregressive generation:
        - Prefill: [B, T, D] where T is prompt length - we steer position T-1
        - Decoding: [B, 1, D] - we steer position 0 (the new token)
        
        Args:
            layer_idx: Index of the layer to hook
            debug: Whether to print debug information
            
        Returns:
            Hook function
        """
        module = self.steering_modules[layer_idx]
        probe = self.risk_probes[layer_idx]
        
        call_count = [0]  # Use list to allow mutation in closure
        
        def hook_fn(module_self, input, output):
            call_count[0] += 1
            
            # Get activation
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            # activation shape: [batch, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = activation.shape
            
            # CRITICAL FIX: Always steer the last position
            # For prefill: this is the last prompt token
            # For decoding: this is the newly generated token (seq_len=1)
            last_idx = seq_len - 1
            
            # Get the last token's activation
            last_token_act = activation[:, last_idx, :]  # [batch, hidden_dim]
            
            # Get the device of the activation
            act_device = last_token_act.device
            
            # Move probe and module to activation device if needed
            probe_device = next(probe.parameters()).device
            if probe_device != act_device:
                probe_to_use = probe.to(act_device)
            else:
                probe_to_use = probe
            
            module_device = next(module.parameters()).device
            if module_device != act_device:
                module_to_use = module.to(act_device)
            else:
                module_to_use = module
            
            # Compute risk score
            with torch.no_grad():
                risk_score = probe_to_use(last_token_act.float()).squeeze(-1)  # [batch]
            
            # Store original activation for comparison
            original_act = last_token_act.clone()
            
            # Apply steering using module's method
            steered_act = module_to_use.get_steered_activation(
                last_token_act.float(), 
                risk_score
            )
            
            # Compute the actual change (delta)
            delta = steered_act - original_act
            delta_norm = delta.norm(dim=-1).mean().item()
            
            # Debug: Print steering info for first few calls
            if debug and call_count[0] <= 5:
                gate = module_to_use.compute_gate(risk_score)
                print(f"  [Layer {layer_idx}] Call {call_count[0]}: "
                      f"seq_len={seq_len}, last_idx={last_idx}, "
                      f"risk={risk_score.mean().item():.4f}, "
                      f"gate={gate.mean().item():.4f}, "
                      f"delta_norm={delta_norm:.6f}")
            
            # Replace ONLY the last token activation with steered version
            activation[:, last_idx, :] = steered_act.to(activation.dtype)
            
            if isinstance(output, tuple):
                return (activation, *output[1:])
            else:
                return activation
        
        return hook_fn
    
    def register_hooks(self, model):
        """
        Register steering hooks on the model.
        
        Args:
            model: The model to register hooks on
            
        Returns:
            List of hook handles
        """
        handles = []
        
        for layer_idx in self.layer_indices:
            layer = model.model.layers[layer_idx]
            hook = self.create_steering_hook(layer_idx)
            handle = layer.register_forward_hook(hook)
            handles.append(handle)
        
        return handles
    
    def remove_hooks(self, handles: List):
        """Remove all registered hooks."""
        for handle in handles:
            handle.remove()


def evaluate_steering_unlearn(
    cfg,
    model_base,
    steering_modules: Dict[int, nn.Module],
    risk_probes: Dict[int, nn.Module],
    layer_indices: List[int],
    data_path: str,
    device: str = "cuda",
    compute_es_score: bool = True,
) -> Dict:
    """
    Evaluate the steering unlearn method.
    
    This function reuses LUNAR's evaluation pipeline, applying steering
    hooks during generation.
    
    Args:
        cfg: Configuration object
        model_base: The model wrapper
        steering_modules: Trained steering modules
        risk_probes: Trained risk probes
        layer_indices: Target layer indices
        data_path: Path to evaluation data
        device: Device to use
        compute_es_score: Whether to compute ES score
        
    Returns:
        Dictionary of evaluation results
    """
    print(f"\n{'='*60}")
    print("Evaluation Phase")
    print(f"{'='*60}")
    
    # Create hook manager
    hook_manager = SteeringHookManager(
        steering_modules=steering_modules,
        risk_probes=risk_probes,
        layer_indices=layer_indices,
        device=device,
    )
    
    # Prepare hooks (enable debug for first layer)
    fwd_hooks = []
    for i, layer_idx in enumerate(layer_indices):
        # Enable debug for the first layer to see steering effect
        hook_fn = hook_manager.create_steering_hook(layer_idx, debug=(i == 0))
        # Try different layer access patterns
        if hasattr(model_base, 'model'):
            if hasattr(model_base.model, 'layers'):
                layer = model_base.model.layers[layer_idx]
            elif hasattr(model_base.model, 'model') and hasattr(model_base.model.model, 'layers'):
                layer = model_base.model.model.layers[layer_idx]
            else:
                raise AttributeError(f"Cannot find layers in model_base.model")
        else:
            raise AttributeError(f"model_base has no 'model' attribute")
        fwd_hooks.append((layer, hook_fn))
    
    # Evaluate on forget set
    print("\nEvaluating on forget set...")
    eval_logs_forget = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=model_base.tokenizer,
        model=model_base,
        eval_target="forget_edge",
        fwd_pre_hooks=[],
        fwd_hooks=fwd_hooks,
        output_es_score=compute_es_score,
    )
    
    # Evaluate on retain set
    print("\nEvaluating on retain set...")
    eval_logs_retain = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=model_base.tokenizer,
        model=model_base,
        eval_target="retained_edge",
        fwd_pre_hooks=[],
        fwd_hooks=fwd_hooks,
        output_es_score=False,
    )
    
    # Evaluate on factual data if configured
    eval_logs_factual = None
    if cfg.if_eval_factual:
        print("\nEvaluating on factual data...")
        eval_logs_factual = custom_evaluate(
            cfg=cfg,
            data_path=cfg.factual_data_path,
            tokenizer=model_base.tokenizer,
            model=model_base,
            eval_target="factual_data",
            fwd_pre_hooks=[],
            fwd_hooks=fwd_hooks,
            output_es_score=False,
        )
    
    # Compile results
    results = {
        "forget": eval_logs_forget,
        "retain": eval_logs_retain,
    }
    
    if eval_logs_factual:
        results["factual"] = eval_logs_factual
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Results Summary")
    print("="*60)
    
    print("\nForget Set:")
    print(f"  ROUGE-1 Recall: {eval_logs_forget.get('rouge1_recall', 'N/A'):.4f}" if isinstance(eval_logs_forget.get('rouge1_recall'), float) else f"  ROUGE-1 Recall: {eval_logs_forget.get('rouge1_recall', 'N/A')}")
    print(f"  ROUGE-L Recall: {eval_logs_forget.get('rougeL_recall', 'N/A'):.4f}" if isinstance(eval_logs_forget.get('rougeL_recall'), float) else f"  ROUGE-L Recall: {eval_logs_forget.get('rougeL_recall', 'N/A')}")
    if compute_es_score and 'es_score' in eval_logs_forget:
        es = eval_logs_forget['es_score']
        print(f"  ES Score: {es:.4f}" if isinstance(es, float) else f"  ES Score: {es}")
    
    print("\nRetain Set:")
    print(f"  ROUGE-1 Recall: {eval_logs_retain.get('rouge1_recall', 'N/A'):.4f}" if isinstance(eval_logs_retain.get('rouge1_recall'), float) else f"  ROUGE-1 Recall: {eval_logs_retain.get('rouge1_recall', 'N/A')}")
    print(f"  ROUGE-L Recall: {eval_logs_retain.get('rougeL_recall', 'N/A'):.4f}" if isinstance(eval_logs_retain.get('rougeL_recall'), float) else f"  ROUGE-L Recall: {eval_logs_retain.get('rougeL_recall', 'N/A')}")
    
    if eval_logs_factual:
        print("\nFactual Data:")
        print(f"  ROUGE-1 Recall: {eval_logs_factual.get('rouge1_recall', 'N/A'):.4f}" if isinstance(eval_logs_factual.get('rouge1_recall'), float) else f"  ROUGE-1 Recall: {eval_logs_factual.get('rouge1_recall', 'N/A')}")
        print(f"  ROUGE-L Recall: {eval_logs_factual.get('rougeL_recall', 'N/A'):.4f}" if isinstance(eval_logs_factual.get('rougeL_recall'), float) else f"  ROUGE-L Recall: {eval_logs_factual.get('rougeL_recall', 'N/A')}")
    
    return results


def compute_additional_metrics(
    model_base,
    steering_modules: Dict[int, nn.Module],
    risk_probes: Dict[int, nn.Module],
    cone_targets: Dict[int, Dict],
    forget_instructions: List[str],
    retain_instructions: List[str],
    layer_indices: List[int],
    device: str = "cuda",
    batch_size: int = 4,
) -> Dict:
    """
    Compute additional diagnostic metrics.
    
    This includes:
    - Average risk before/after steering
    - Average cosine similarity to target before/after steering
    
    Args:
        model_base: The model wrapper
        steering_modules: Trained steering modules
        risk_probes: Trained risk probes
        cone_targets: Cone targets
        forget_instructions: Forget set instructions
        retain_instructions: Retain set instructions
        layer_indices: Target layer indices
        device: Device to use
        batch_size: Batch size for processing
        
    Returns:
        Dictionary of additional metrics
    """
    import torch.nn.functional as F
    from src.hook_for_unlearn import get_post_block_activation
    
    print("\nComputing additional diagnostic metrics...")
    
    metrics = {
        'forget': {},
        'retain': {},
    }
    
    # Extract activations for a sample of instructions
    n_samples = min(len(forget_instructions), 50)
    forget_sample = forget_instructions[:n_samples]
    retain_sample = retain_instructions[:n_samples]
    
    for layer_idx in layer_indices:
        # Get activations for forget set
        forget_acts = get_post_block_activation(
            model=model_base.model,
            input_data=forget_sample,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            layer_idx=layer_idx,
            batch_size=batch_size,
        )
        forget_acts_tensor = torch.cat([a[:, -1, :] for a in forget_acts], dim=0)
        
        # Get activations for retain set
        retain_acts = get_post_block_activation(
            model=model_base.model,
            input_data=retain_sample,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            layer_idx=layer_idx,
            batch_size=batch_size,
        )
        retain_acts_tensor = torch.cat([a[:, -1, :] for a in retain_acts], dim=0)
        
        # Compute risk scores before steering
        probe = risk_probes[layer_idx]
        with torch.no_grad():
            forget_risk_before = probe(forget_acts_tensor.to(device)).mean().item()
            retain_risk_before = probe(retain_acts_tensor.to(device)).mean().item()
        
        # Compute cosine similarity to target before steering
        target_axis = cone_targets[layer_idx]['ignorance']
        if hasattr(target_axis, 'axis'):
            axis = target_axis.axis.to(device)
        else:
            axis = torch.tensor(target_axis['axis']).to(device)
        
        forget_acts_norm = F.normalize(forget_acts_tensor.to(device), dim=-1)
        axis_norm = F.normalize(axis.unsqueeze(0), dim=-1).squeeze(0)
        forget_cosim_before = (forget_acts_norm @ axis_norm).mean().item()
        
        retain_acts_norm = F.normalize(retain_acts_tensor.to(device), dim=-1)
        retain_cosim_before = (retain_acts_norm @ axis_norm).mean().item()
        
        metrics['forget'][f'layer{layer_idx}'] = {
            'risk_before': forget_risk_before,
            'cosim_to_target_before': forget_cosim_before,
        }
        metrics['retain'][f'layer{layer_idx}'] = {
            'risk_before': retain_risk_before,
            'cosim_to_target_before': retain_cosim_before,
        }
    
    return metrics


def save_evaluation_results(
    results: Dict,
    additional_metrics: Optional[Dict],
    save_dir: str,
    experiment_name: str = "steering_unlearn",
):
    """
    Save evaluation results to disk.
    
    Args:
        results: Main evaluation results
        additional_metrics: Additional diagnostic metrics
        save_dir: Directory to save results
        experiment_name: Name of the experiment
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save main results
    results_path = os.path.join(save_dir, f"{experiment_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results to {results_path}")
    
    # Save additional metrics
    if additional_metrics:
        metrics_path = os.path.join(save_dir, f"{experiment_name}_additional_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(additional_metrics, f, indent=2, default=str)
        print(f"Saved additional metrics to {metrics_path}")
    
    # Create summary table
    summary = {
        "method": experiment_name,
        "forget_rouge1": results.get('forget', {}).get('rouge1_recall', 'N/A'),
        "retain_rouge1": results.get('retain', {}).get('rouge1_recall', 'N/A'),
        "forget_rougeL": results.get('forget', {}).get('rougeL_recall', 'N/A'),
        "retain_rougeL": results.get('retain', {}).get('rougeL_recall', 'N/A'),
    }
    
    if 'es_score' in results.get('forget', {}):
        summary['es_score'] = results['forget']['es_score']
    
    summary_path = os.path.join(save_dir, f"{experiment_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    # Print markdown table
    print("\n" + "="*60)
    print("Results Table (Markdown)")
    print("="*60)
    print("| Metric | Forget | Retain |")
    print("|--------|--------|--------|")
    print(f"| ROUGE-1 | {summary['forget_rouge1']:.4f} | {summary['retain_rouge1']:.4f} |" if isinstance(summary['forget_rouge1'], float) else f"| ROUGE-1 | {summary['forget_rouge1']} | {summary['retain_rouge1']} |")
    print(f"| ROUGE-L | {summary['forget_rougeL']:.4f} | {summary['retain_rougeL']:.4f} |" if isinstance(summary['forget_rougeL'], float) else f"| ROUGE-L | {summary['forget_rougeL']} | {summary['retain_rougeL']} |")


if __name__ == "__main__":
    print("Steering Unlearn Evaluator Module")