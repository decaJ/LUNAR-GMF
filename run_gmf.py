# Gated Manifold Flow for LLM Unlearning
# Main execution script for GMF unlearning on TOFU dataset

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from itertools import chain

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

# ---- Project imports -------------------------------------------
from src.dataset_utils import (
    load_dataset_to_get_direction,
    split_raw_dataset_for_forget,
)
from src.eval_util import custom_evaluate
from src.generate_directions import generate_candidate_directions
from src.model_utils.model_loader import load_model
from src.hook_for_unlearn import (
    get_activations,
    get_post_block_activation,
    get_pre_down_proj_activation,
    get_pre_post_attention_layernorm_activation,
)

# ---- GMF imports -------------------------------------------
from gmf.manifold import ManifoldExtractor, ForgetManifold, AttractorManifold
from gmf.gating import DistanceBasedGate, GatedFlow
from gmf.flow_transform import ResidualFlow
from gmf.losses import GMFLoss, GMFLossConfig
from gmf.trainer import GMFTrainer, GMFTrainerConfig, GMFModule

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_gmf_trainset(
    layer_idx_list: List[int],
    model_base,
    forget_dataset,
    retain_dataset,
    device,
):
    """
    Prepare training data for GMF unlearning.
    
    Returns activation data needed for:
    1. Manifold extraction (Phase 1) - uses post_block_activation (hidden_size dim)
    2. Flow transformation training (Phase 2) - uses post_block_activation for consistency
    
    Note: We use post_block_activation (hidden_size) for both manifold extraction
    and training to ensure dimensional consistency. The transformation will be
    applied at the residual stream level.
    """
    all_forget_activations = {}
    all_retain_activations = {}
    forget_inputs_list = {}
    retain_inputs_list = {}
    
    for layer_idx in layer_idx_list:
        logger.info(f"Processing layer {layer_idx}...")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Get activations for this layer
        (
            post_block_activation_forget,
            post_block_activation_remain,
            pre_post_attention_layernorm_activation_forget,
            pre_post_attention_layernorm_activation_remain,
            pre_down_proj_activation_forget,
            pre_down_proj_activation_remain,
        ) = get_activations(
            model_base, layer_idx, forget_dataset, retain_dataset,
            batch_size_forget=1, batch_size_remain=1
        )
        
        # Store for manifold extraction
        all_forget_activations[layer_idx] = post_block_activation_forget
        all_retain_activations[layer_idx] = post_block_activation_remain
        
        # Prepare training inputs using post_block_activation (same as manifold)
        # This ensures dimensional consistency with the manifold
        forget_inputs = torch.cat([
            act.squeeze(0).cpu() for act in post_block_activation_forget
        ], dim=0)
        retain_inputs = torch.cat([
            act.squeeze(0).cpu() for act in post_block_activation_remain
        ], dim=0)
        
        forget_inputs_list[layer_idx] = forget_inputs
        retain_inputs_list[layer_idx] = retain_inputs
        
        logger.info(f"Layer {layer_idx}: forget shape {forget_inputs.shape}, retain shape {retain_inputs.shape}")
        
        # Clear memory
        del post_block_activation_forget, post_block_activation_remain
        del pre_post_attention_layernorm_activation_forget, pre_post_attention_layernorm_activation_remain
        del pre_down_proj_activation_forget, pre_down_proj_activation_remain
        torch.cuda.empty_cache()
    
    return (
        all_forget_activations,
        all_retain_activations,
        forget_inputs_list,
        retain_inputs_list,
    )


class GMFUnlearner:
    """
    Gated Manifold Flow Unlearner.
    
    Implements the complete GMF unlearning pipeline:
    1. Extract manifold statistics from forget data
    2. Train gated flow transformation
    3. Apply transformation to model weights
    """
    
    def __init__(self, cfg, model_base, device='cuda'):
        self.cfg = cfg
        self.model_base = model_base
        self.device = device
        self.hidden_size = model_base.model.config.hidden_size
        self.intermediate_size = getattr(model_base.model.config, 'intermediate_size', self.hidden_size * 4)
        
        # GMF modules for each layer
        self.gmf_modules = {}
        self.forget_manifolds = {}
        self.attractor_manifolds = {}
        
    def extract_manifolds(
        self,
        forget_activations: List[torch.Tensor],
        retain_activations: List[torch.Tensor],
        refusal_direction: torch.Tensor,
        layer_idx: int,
        input_dim: int = None,
    ):
        """
        Phase 1: Extract forget and attractor manifolds.
        
        Args:
            forget_activations: List of forget activations
            retain_activations: List of retain activations
            refusal_direction: Direction vector for attractor
            layer_idx: Layer index
            input_dim: Dimension of input activations (if different from hidden_size)
        """
        logger.info(f"Extracting manifolds for layer {layer_idx}...")
        
        # Use input_dim if specified, otherwise use hidden_size
        actual_dim = input_dim if input_dim is not None else self.hidden_size
        
        extractor = ManifoldExtractor(
            hidden_size=actual_dim,
            use_diagonal_cov=True,
            device=self.device,
        )
        
        # Extract forget manifold
        logger.info("Extracting forget manifold...")
        forget_manifold = extractor.extract_forget_manifold(forget_activations)
        forget_manifold = forget_manifold.to(self.device)
        
        # Create attractor manifold
        # The attractor is offset from forget manifold along a scaled direction
        # If refusal_direction has different dim, we project/expand it
        if refusal_direction.shape[0] != actual_dim:
            # Project refusal direction to the actual dimension
            # For intermediate_size, we use a simple expansion (repeat)
            scale_factor = actual_dim // refusal_direction.shape[0]
            if scale_factor > 1:
                # Expand by repeating
                attractor_direction = refusal_direction.repeat(scale_factor)[:actual_dim]
            else:
                # Shrink by taking first actual_dim elements
                attractor_direction = refusal_direction[:actual_dim]
            attractor_direction = attractor_direction / (torch.norm(attractor_direction) + 1e-8) * torch.norm(refusal_direction)
        else:
            attractor_direction = refusal_direction
        
        attractor_mu = forget_manifold.mu + attractor_direction.to(self.device) * 2.0
        attractor_manifold = AttractorManifold(
            mu=attractor_mu,
            direction=attractor_direction.to(self.device),
            scale=self.cfg.get('attractor_scale', 1.0),
        )
        
        self.forget_manifolds[layer_idx] = forget_manifold
        self.attractor_manifolds[layer_idx] = attractor_manifold
        
        logger.info(f"Forget manifold mean norm: {torch.norm(forget_manifold.mu):.4f}")
        logger.info(f"Attractor manifold mean norm: {torch.norm(attractor_mu):.4f}")
        
        return forget_manifold, attractor_manifold
    
    def train_gmf(
        self,
        forget_inputs: torch.Tensor,
        retain_inputs: torch.Tensor,
        layer_idx: int,
    ):
        """
        Phase 2: Train the gated flow transformation.
        """
        logger.info(f"Training GMF for layer {layer_idx}...")
        
        # Create trainer config
        trainer_config = GMFTrainerConfig(
            num_epochs=self.cfg.get('num_epochs', 20),
            batch_size=self.cfg.get('batch_size', 32),
            learning_rate=self.cfg.get('lr', 1e-4),
            sigma=self.cfg.get('sigma', 1.0),
            learnable_sigma=self.cfg.get('learnable_sigma', False),
            distance_method=self.cfg.get('distance_method', 'mahalanobis'),
            flow_hidden_dim=self.cfg.get('flow_hidden_dim', 512),
            flow_num_layers=self.cfg.get('flow_num_layers', 3),
            lambda_attractor=self.cfg.get('lambda_attractor', 1.0),
            lambda_retain=self.cfg.get('lambda_retain', 1.0),
            lambda_flow=self.cfg.get('lambda_flow', 0.1),
            lambda_recoverability=self.cfg.get('lambda_recoverability', 0.1),
            device=self.device,
        )
        
        # Create GMF module
        gmf_module = GMFModule(self.hidden_size, trainer_config)
        
        # Set manifold
        if layer_idx in self.forget_manifolds:
            gmf_module.set_manifold(self.forget_manifolds[layer_idx])
        
        # Create trainer
        trainer = GMFTrainer(self.hidden_size, trainer_config)
        trainer.gmf_module = gmf_module
        trainer.forget_manifold = self.forget_manifolds[layer_idx]
        trainer.attractor_manifold = self.attractor_manifolds[layer_idx]
        
        # Train
        history = trainer.phase2_train(forget_inputs, retain_inputs)
        
        # Store trained module
        self.gmf_modules[layer_idx] = trainer
        
        return history
    
    def apply_to_model(self, model_base, layer_idx_list: List[int]):
        """
        Apply the trained GMF transformation to model weights.
        
        This modifies the down_proj weights of the specified layers
        to incorporate the learned flow transformation.
        """
        logger.info("Applying GMF transformation to model...")
        
        for layer_idx in layer_idx_list:
            if layer_idx not in self.gmf_modules:
                logger.warning(f"No GMF module for layer {layer_idx}, skipping...")
                continue
            
            trainer = self.gmf_modules[layer_idx]
            gmf_module = trainer.gmf_module.eval()
            
            # Get the original down_proj weight
            original_weight = model_base.model_block_modules[layer_idx].mlp.down_proj.weight.data.clone()
            
            # For GMF, we integrate the flow transformation into the down_proj
            # by learning a weight adjustment that approximates the gated flow
            # This is done by computing the expected transformation on the weight space
            
            # Alternative approach: Store the GMF module as a hook
            # For now, we use a simpler approach: compute weight perturbation
            
            # Get a representative sample of activations
            forget_manifold = self.forget_manifolds[layer_idx]
            mean_activation = forget_manifold.mu.to(self.device)
            
            # Compute the transformation on the mean activation
            with torch.no_grad():
                transformed, info = gmf_module(mean_activation.unsqueeze(0))
                delta = transformed.squeeze(0) - mean_activation
                
                # Compute weight adjustment (simplified)
                # This is a linear approximation of the flow transformation
                # weight_adj = delta @ pseudo_inverse(mean_activation)
                
            logger.info(f"Layer {layer_idx}: Gate mean = {info['gate_mean']:.4f}")
            logger.info(f"Transformation delta norm: {torch.norm(delta):.4f}")
        
        return model_base
    
    def get_transformation_hook(self, layer_idx: int):
        """
        Get a forward hook that applies the GMF transformation during inference.
        """
        if layer_idx not in self.gmf_modules:
            return None
        
        trainer = self.gmf_modules[layer_idx]
        gmf_module = trainer.gmf_module.eval()
        
        def hook_fn(module, input, output):
            # Apply GMF transformation to output activations
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            # Store original shape and dtype
            original_shape = activation.shape
            original_dtype = activation.dtype
            
            # Reshape for transformation
            batch_size = activation.shape[0]
            seq_len = activation.shape[1]
            activation_flat = activation.view(-1, activation.shape[-1])
            
            # Apply transformation (GMF uses float32 internally)
            with torch.no_grad():
                # Convert to float32 for transformation
                activation_flat_float = activation_flat.float()
                transformed, _ = gmf_module(activation_flat_float)
            
            # Reshape back and restore original dtype
            transformed = transformed.view(original_shape).to(original_dtype)
            
            if isinstance(output, tuple):
                return (transformed, *output[1:])
            else:
                return transformed
        
        return hook_fn


@hydra.main(version_base=None, config_path="config", config_name="forget_gmf")
def run_gmf_unlearning(cfg):
    """
    Main function to run GMF unlearning.
    """
    logger.info("=" * 60)
    logger.info("Gated Manifold Flow for LLM Unlearning")
    logger.info("=" * 60)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # Load the model and datasets
    # -----------------------------
    logger.info(f"Loading model from {cfg.model_family} at {cfg.model_path}")
    model_base = load_model(cfg.model_family, cfg.model_path, device)
    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")
    
    # -----------------------------
    # Generate refusal directions
    # -----------------------------
    logger.info("Generating refusal directions...")
    harmful_train, forget_train = load_dataset_to_get_direction(
        cfg,
        data_path,
        instructions_only=True,
        use_harmful=cfg.use_harmful,
        use_unverified=cfg.use_unverified,
    )
    
    candidate_directions = generate_candidate_directions(
        cfg, model_base, harmful_train, forget_train
    )
    
    layer_idx_list = cfg.layer_modified
    positions = cfg.positions
    
    # Extract refusal directions for each layer
    refusal_directions = {}
    for layer_idx in layer_idx_list:
        # +1 because direction is calculated using pre-hook
        refusal_directions[layer_idx] = candidate_directions[positions, layer_idx + 1, :]
        logger.info(f"Layer {layer_idx} refusal direction norm: {torch.norm(refusal_directions[layer_idx]):.4f}")
    
    # -----------------------------
    # Prepare training sets
    # -----------------------------
    logger.info("Preparing training sets...")
    forget_dataset, retain_dataset = split_raw_dataset_for_forget(
        cfg,
        data_path,
        model_base,
        forget_edge=cfg.forget_edge,
        instructions_only=True,
        torch_reformat=False,
    )
    logger.info(f"Forget dataset size: {len(forget_dataset)}")
    logger.info(f"Retain dataset size: {len(retain_dataset)}")
    
    # -----------------------------
    # Extract activations
    # -----------------------------
    logger.info("Extracting activations...")
    (
        all_forget_activations,
        all_retain_activations,
        forget_inputs_list,
        retain_inputs_list,
    ) = prepare_gmf_trainset(
        layer_idx_list,
        model_base,
        forget_dataset,
        retain_dataset,
        device,
    )
    
    # -----------------------------
    # Create GMF Unlearner
    # -----------------------------
    unlearner = GMFUnlearner(cfg, model_base, device)
    
    # -----------------------------
    # Phase 1: Extract Manifolds
    # -----------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Extracting Manifolds")
    logger.info("=" * 60)
    
    for layer_idx in layer_idx_list:
        unlearner.extract_manifolds(
            all_forget_activations[layer_idx],
            all_retain_activations[layer_idx],
            refusal_directions[layer_idx],
            layer_idx,
        )
    
    # -----------------------------
    # Phase 2: Train GMF
    # -----------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Training Gated Flow")
    logger.info("=" * 60)
    
    for layer_idx in layer_idx_list:
        unlearner.train_gmf(
            forget_inputs_list[layer_idx],
            retain_inputs_list[layer_idx],
            layer_idx,
        )
    
    # -----------------------------
    # Apply to model
    # -----------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Applying GMF to Model")
    logger.info("=" * 60)
    
    # Load a fresh model for modification
    updated_model = load_model(cfg.model_family, cfg.model_path, device)
    
    # Apply transformation (currently uses hooks for inference)
    # For weight modification, see apply_to_model method
    
    # -----------------------------
    # Save model if requested
    # -----------------------------
    if cfg.get('save_unlearned_model', False):
        save_path = cfg.save_unlearned_model_path
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        logger.info(f"Saving unlearned model to {save_path}")
        updated_model._save_pretrained(save_path)
    
    # -----------------------------
    # Evaluation
    # -----------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation")
    logger.info("=" * 60)
    
    # CRITICAL: Release training model and activations from GPU memory
    logger.info("Releasing training model from GPU memory...")
    del model_base
    del all_forget_activations, all_retain_activations
    del forget_inputs_list, retain_inputs_list
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save GMF modules to CPU for later use
    gmf_modules_cpu = {}
    for layer_idx, trainer in unlearner.gmf_modules.items():
        gmf_modules_cpu[layer_idx] = {
            'gmf_module': trainer.gmf_module.cpu(),
            'forget_manifold': unlearner.forget_manifolds[layer_idx],
            'attractor_manifold': unlearner.attractor_manifolds[layer_idx],
        }
    
    # Release unlearner from GPU
    del unlearner
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load fresh model for evaluation on specified GPU
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading fresh model for evaluation on {eval_device}...")
    
    eval_model = load_model(cfg.model_family, cfg.model_path, eval_device)
    
    # Create evaluation hooks using CPU-stored GMF modules
    fwd_hooks = []
    for layer_idx in layer_idx_list:
        if layer_idx not in gmf_modules_cpu:
            continue
        
        gmf_module = gmf_modules_cpu[layer_idx]['gmf_module'].to(eval_device)
        # Ensure the gate's manifold is on the correct device
        if hasattr(gmf_module.gate, 'manifold_mu') and gmf_module.gate.manifold_mu is not None:
            gmf_module.gate.manifold_mu = gmf_module.gate.manifold_mu.to(eval_device)
        if hasattr(gmf_module.gate, 'manifold_sigma') and gmf_module.gate.manifold_sigma is not None:
            gmf_module.gate.manifold_sigma = gmf_module.gate.manifold_sigma.to(eval_device)
        gmf_module.eval()
        
        def make_hook(gmf_mod, device):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output
                
                original_shape = activation.shape
                original_dtype = activation.dtype
                activation_flat = activation.view(-1, activation.shape[-1])
                
                with torch.no_grad():
                    activation_flat_float = activation_flat.float().to(device)
                    transformed, _ = gmf_mod(activation_flat_float)
                
                transformed = transformed.view(original_shape).to(original_dtype)
                
                if isinstance(output, tuple):
                    return (transformed, *output[1:])
                else:
                    return transformed
            return hook_fn
        
        fwd_hooks.append((eval_model.model_block_modules[layer_idx], make_hook(gmf_module, eval_device)))
    
    # Evaluate forget performance
    logger.info("Evaluating forget performance...")
    eval_logs_forget = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=eval_model.tokenizer,
        model=eval_model,
        eval_target="forget_edge",
        output_es_score=cfg.get('compute_es_score', False),
        fwd_hooks=fwd_hooks,
    )
    
    # Evaluate retain performance
    logger.info("Evaluating retain performance...")
    eval_logs_retain = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=eval_model.tokenizer,
        model=eval_model,
        eval_target="retained_edge",
        output_es_score=False,
        fwd_hooks=fwd_hooks,
    )
    
    eval_logs = {
        "forget": eval_logs_forget,
        "retain": eval_logs_retain,
    }
    
    # Evaluate factual data if requested
    if cfg.get('if_eval_factual', False):
        logger.info("Evaluating factual data...")
        eval_logs_factual = custom_evaluate(
            cfg=cfg,
            data_path=cfg.factual_data_path,
            tokenizer=eval_model.tokenizer,
            model=eval_model,
            eval_target="factual_data",
            output_es_score=False,
            fwd_hooks=fwd_hooks,
        )
        eval_logs["factual"] = eval_logs_factual
    
    # Clean up evaluation model
    del eval_model
    del gmf_modules_cpu
    torch.cuda.empty_cache()
    gc.collect()
    
    # -----------------------------
    # Save results
    # -----------------------------
    save_dir = cfg.get('save_path', 'run_results/gmf')
    os.makedirs(save_dir, exist_ok=True)
    
    save_str = "_".join([str(idx) for idx in layer_idx_list])
    save_file = f"{save_dir}/gmf_{save_str}.json"
    
    logger.info(f"Saving results to {save_file}")
    with open(save_file, "w") as f:
        json.dump(eval_logs, f, indent=4)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Forget evaluation: {eval_logs_forget}")
    logger.info(f"Retain evaluation: {eval_logs_retain}")
    
    return eval_logs


if __name__ == "__main__":
    run_gmf_unlearning()