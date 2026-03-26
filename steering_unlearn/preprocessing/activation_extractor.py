# Copyright (c) Steering Unlearn Project
"""
Activation Extractor for Steering Unlearn Method

This module extracts activations from target layers for:
- forget set
- retain set
- ignorance reference set
- refusal reference set
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import from LUNAR
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.hook_for_unlearn import (
    add_hooks,
    get_activations_pre_hook,
    get_activations_fwd_hook,
)


@dataclass
class ActivationConfig:
    """Configuration for activation extraction"""
    batch_size: int = 4
    positions: int = -1  # last token position
    move_to_cpu: bool = True


def extract_activations_for_layer(
    model_base,
    layer_idx: int,
    instructions: List[str],
    config: ActivationConfig,
    hook_type: str = "post_block",  # "post_block" or "pre_down_proj"
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Extract activations from a specific layer for given instructions.
    
    Args:
        model_base: The model wrapper (LUNAR model base)
        layer_idx: Index of the layer to extract from
        instructions: List of instruction strings
        config: Activation extraction configuration
        hook_type: Type of hook to use for extraction
        
    Returns:
        Tuple of (activations, attention_masks) - lists of tensors per batch
    """
    torch.cuda.empty_cache()
    activations = []
    attention_masks = []  # Store attention masks for each batch
    
    # Use model_block_modules from LUNAR model base
    layers = model_base.model_block_modules
    
    if hook_type == "post_block":
        # Hook after the transformer block (residual stream output)
        hook_module = layers[layer_idx]
        hook_fn = get_activations_fwd_hook(
            cache=activations, 
            move_to_cpu=config.move_to_cpu
        )
        fwd_hooks = [(hook_module, hook_fn)]
        pre_hooks = []
    elif hook_type == "pre_down_proj":
        # Hook before down_proj (MLP output)
        hook_module = layers[layer_idx].mlp.down_proj
        hook_fn = get_activations_pre_hook(
            cache=activations,
            move_to_cpu=config.move_to_cpu
        )
        pre_hooks = [(hook_module, hook_fn)]
        fwd_hooks = []
    else:
        raise ValueError(f"Unknown hook_type: {hook_type}")
    
    # Process in batches
    for i in tqdm(range(0, len(instructions), config.batch_size), 
                  desc=f"Extracting layer {layer_idx}"):
        batch_instructions = instructions[i:i + config.batch_size]
        inputs = model_base.tokenize_instructions_fn(instructions=batch_instructions)
        
        # Store attention mask for this batch
        attention_masks.append(inputs.attention_mask.cpu())
        
        with add_hooks(
            module_forward_pre_hooks=pre_hooks,
            module_forward_hooks=fwd_hooks
        ):
            with torch.no_grad():
                model_base.model(
                    input_ids=inputs.input_ids.to(model_base.model.device),
                    attention_mask=inputs.attention_mask.to(model_base.model.device),
                )
        
        torch.cuda.empty_cache()
    
    return activations, attention_masks


def extract_last_token_activations(
    activations: List[torch.Tensor],
    attention_masks: Optional[List[torch.Tensor]] = None,
    positions: int = -1,
) -> torch.Tensor:
    """
    Extract the activation at the specified position (default: last token).
    
    If attention_masks is provided, uses it to find the last valid token
    for each sample (ignoring padding).
    
    Args:
        activations: List of activation tensors [batch, seq_len, hidden_dim]
        attention_masks: Optional list of attention mask tensors [batch, seq_len]
        positions: Position index (negative indices supported, ignored if attention_masks provided)
        
    Returns:
        Tensor of shape [num_samples, hidden_dim]
    """
    extracted = []
    
    for idx, act in enumerate(activations):
        # act shape: [batch, seq_len, hidden_dim]
        if act.dim() == 3:
            batch_size = act.shape[0]
            
            if attention_masks is not None and idx < len(attention_masks):
                # Use attention mask to find last valid token
                attn_mask = attention_masks[idx]  # [batch, seq_len]
                
                for i in range(batch_size):
                    # Find the position of the last valid token
                    valid_positions = attn_mask[i].nonzero(as_tuple=True)[0]
                    if len(valid_positions) > 0:
                        last_valid_idx = valid_positions[-1].item()
                    else:
                        # Fallback to last position if no valid tokens
                        last_valid_idx = -1
                    
                    extracted.append(act[i, last_valid_idx, :].cpu())
            else:
                # No attention mask, use simple indexing
                for i in range(batch_size):
                    extracted.append(act[i, positions, :].cpu())
                    
        elif act.dim() == 2:
            extracted.append(act[positions, :].cpu())
        else:
            raise ValueError(f"Unexpected activation shape: {act.shape}")
    
    return torch.stack(extracted, dim=0)


def extract_all_layer_activations(
    model_base,
    layer_indices: List[int],
    forget_instructions: List[str],
    retain_instructions: List[str],
    ignorance_instructions: List[str],
    refusal_instructions: List[str],
    config: ActivationConfig,
    save_dir: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Extract activations from all target layers for all data types.
    
    Uses attention masks to correctly identify the last valid token
    (ignoring padding tokens).
    
    Args:
        model_base: The model wrapper
        layer_indices: List of layer indices to extract from
        forget_instructions: Forget set instructions
        retain_instructions: Retain set instructions
        ignorance_instructions: Ignorance reference instructions
        refusal_instructions: Refusal reference instructions
        config: Activation extraction configuration
        save_dir: Directory to save activations (optional)
        debug: If True, print debug info for first few samples
        
    Returns:
        Dictionary with activations organized by data type and layer
    """
    all_activations = {
        'forget': {},
        'retain': {},
        'ignorance': {},
        'refusal': {},
    }
    
    data_types = {
        'forget': forget_instructions,
        'retain': retain_instructions,
        'ignorance': ignorance_instructions,
        'refusal': refusal_instructions,
    }
    
    for layer_idx in layer_indices:
        print(f"\n{'='*50}")
        print(f"Processing Layer {layer_idx}")
        print(f"{'='*50}")
        
        for data_type, instructions in data_types.items():
            print(f"\nExtracting {data_type} activations...")
            
            # Extract raw activations and attention masks
            raw_activations, attention_masks = extract_activations_for_layer(
                model_base=model_base,
                layer_idx=layer_idx,
                instructions=instructions,
                config=config,
                hook_type="post_block",  # Use post-block for residual stream
            )
            
            # Debug: Print info for first few samples
            if debug and data_type == 'forget':
                print(f"\n  Debug info for first 3 samples:")
                for batch_idx in range(min(3, len(raw_activations))):
                    act = raw_activations[batch_idx]
                    attn = attention_masks[batch_idx]
                    for sample_idx in range(min(2, act.shape[0])):
                        seq_len = act.shape[1]
                        valid_len = attn[sample_idx].sum().item()
                        last_valid = attn[sample_idx].nonzero(as_tuple=True)[0][-1].item()
                        print(f"    Sample {batch_idx}_{sample_idx}: seq_len={seq_len}, "
                              f"valid_tokens={valid_len}, last_valid_idx={last_valid}")
            
            # Extract last token activations using attention masks
            last_token_acts = extract_last_token_activations(
                raw_activations,
                attention_masks=attention_masks,  # Use attention masks!
                positions=config.positions,
            )
            
            all_activations[data_type][layer_idx] = last_token_acts
            
            # Save to disk if specified
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(
                    save_dir, 
                    f"{data_type}_acts_layer{layer_idx}.pt"
                )
                torch.save(last_token_acts, save_path)
                print(f"Saved to {save_path}")
            
            # Clear memory
            del raw_activations
            del attention_masks
            torch.cuda.empty_cache()
    
    return all_activations


def load_instructions_from_json(file_path: str, key: str = "instruction") -> List[str]:
    """Load instructions from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Check if it's a list of dicts with 'instruction' or 'question' key
        if isinstance(data[0], dict):
            if 'instruction' in data[0]:
                return [item['instruction'] for item in data]
            elif 'question' in data[0]:
                return [item['question'] for item in data]
        else:
            # List of strings
            return data
    else:
        raise ValueError(f"Unexpected data format in {file_path}")


def prepare_tofu_instructions(
    data_path: str,
    forget_edges: List[str],
    include_answers: bool = False,
) -> tuple:
    """
    Prepare forget and retain instructions from TOFU dataset.
    
    Args:
        data_path: Path to TOFU JSON file
        forget_edges: List of edges (authors) to forget
        include_answers: If True, also return answers (for forget set)
        
    Returns:
        If include_answers=False:
            Tuple of (forget_instructions, retain_instructions)
        If include_answers=True:
            Tuple of (forget_instructions, forget_answers, retain_instructions)
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    forget_instructions = []
    forget_answers = []
    retain_instructions = []
    
    for item in data:
        question = item.get('question', item.get('instruction', ''))
        edge = item.get('edge', '')
        
        if edge in forget_edges:
            forget_instructions.append(question)
            if 'answer' in item:
                forget_answers.append(item['answer'])
        else:
            retain_instructions.append(question)
    
    print(f"Forget set size: {len(forget_instructions)}")
    print(f"Retain set size: {len(retain_instructions)}")
    
    if include_answers:
        return forget_instructions, forget_answers, retain_instructions
    else:
        return forget_instructions, retain_instructions


def load_or_extract_activations(
    model_base,
    cfg,
    layer_indices: List[int],
    save_dir: str,
    force_extract: bool = False,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Load activations from disk if available, otherwise extract and save.
    
    Args:
        model_base: The model wrapper
        cfg: Configuration object
        layer_indices: List of layer indices
        save_dir: Directory for saving/loading activations
        force_extract: Force re-extraction even if files exist
        
    Returns:
        Dictionary with activations organized by data type and layer
    """
    act_config = ActivationConfig(
        batch_size=cfg.activation_batch_size,
        positions=cfg.activation_positions,
        move_to_cpu=True,
    )
    
    # Check if all files exist
    all_exist = True
    for layer_idx in layer_indices:
        for data_type in ['forget', 'retain', 'ignorance', 'refusal']:
            path = os.path.join(save_dir, f"{data_type}_acts_layer{layer_idx}.pt")
            if not os.path.exists(path):
                all_exist = False
                break
        if not all_exist:
            break
    
    if all_exist and not force_extract:
        print("Loading activations from disk...")
        all_activations = {
            'forget': {},
            'retain': {},
            'ignorance': {},
            'refusal': {},
        }
        
        for layer_idx in layer_indices:
            for data_type in ['forget', 'retain', 'ignorance', 'refusal']:
                path = os.path.join(save_dir, f"{data_type}_acts_layer{layer_idx}.pt")
                all_activations[data_type][layer_idx] = torch.load(path)
        
        return all_activations
    
    # Need to extract activations
    print("Extracting activations...")
    
    # Load TOFU data
    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")
    forget_instructions, retain_instructions = prepare_tofu_instructions(
        data_path=data_path,
        forget_edges=cfg.forget_edge,
    )
    
    # Load reference data
    ignorance_instructions = load_instructions_from_json(cfg.ignorance_reference_path)
    refusal_instructions = load_instructions_from_json(cfg.refusal_reference_path)
    
    # Extract activations
    return extract_all_layer_activations(
        model_base=model_base,
        layer_indices=layer_indices,
        forget_instructions=forget_instructions,
        retain_instructions=retain_instructions,
        ignorance_instructions=ignorance_instructions,
        refusal_instructions=refusal_instructions,
        config=act_config,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    # Test the activation extractor
    print("Activation Extractor Module")