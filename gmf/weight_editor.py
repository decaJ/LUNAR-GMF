# GMF v2 - Subspace-Projected Weight Editing (SPWE)
#
# Key idea:
#   Forget knowledge is encoded in specific low-dimensional directions of the
#   residual stream (the "forget subspace"). These directions appear in the
#   output of MLP down_proj and attention o_proj at the target layer.
#
#   By projecting out the forget-specific subspace from these weight matrices,
#   the layer can no longer encode or retrieve forget knowledge — permanently,
#   with no inference overhead and no gate required.
#
# Weight edit formula:
#   W_new = W - lambda * U @ U^T @ W
#         = (I - lambda * U @ U^T) @ W
#
#   where U is the (d_model, k) forget-specific PCA basis (retain directions removed).
#   lambda in [0, 1]: erasure strength. lambda=1 = complete projection out.
#
# Comparison to inference-time hook:
#   Hook: modify activation at every forward pass; requires gate to distinguish forget/retain
#   SPWE: modify weights once; no gate needed; permanent and efficient

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# Subspace extraction
# ======================================================================

def compute_forget_specific_subspace(
    forget_activations: torch.Tensor,  # (N_f, d_model)
    retain_activations: torch.Tensor,  # (N_r, d_model)
    k: int = 5,
    retain_k: int = 10,
) -> torch.Tensor:
    """
    Compute the forget-specific subspace: directions that appear in forget
    activations but NOT in retain activations.

    Steps:
        1. PCA on forget activations  -> U_f  (d_model, k)
        2. PCA on retain activations  -> U_r  (d_model, retain_k)
        3. Project out retain directions from U_f:
               U_specific = (I - U_r @ U_r^T) @ U_f
        4. Re-orthonormalise via QR

    Args:
        forget_activations: last-token activations for forget set
        retain_activations: last-token activations for retain set
        k:        number of forget PCA components
        retain_k: number of retain PCA components to suppress

    Returns:
        U_erase: (d_model, k_eff) orthonormal forget-specific directions
    """
    forget_activations = forget_activations.float().cpu()
    retain_activations = retain_activations.float().cpu()

    d_model = forget_activations.shape[1]

    # --- Step 1: forget PCA ---
    mu_f = forget_activations.mean(dim=0)
    centered_f = forget_activations - mu_f
    k_f = min(k, centered_f.shape[0] - 1, d_model)
    _, S_f, Vh_f = torch.linalg.svd(centered_f, full_matrices=False)
    U_f = Vh_f[:k_f, :].T.contiguous()  # (d_model, k_f)
    var_f = (S_f[:k_f] ** 2) / (S_f ** 2).sum()
    logger.info(f"  Forget PCA: k={k_f}, top-3 var={var_f[:3].tolist()}")

    # --- Step 2: retain PCA ---
    mu_r = retain_activations.mean(dim=0)
    centered_r = retain_activations - mu_r
    k_r = min(retain_k, centered_r.shape[0] - 1, d_model)
    _, _, Vh_r = torch.linalg.svd(centered_r, full_matrices=False)
    U_r = Vh_r[:k_r, :].T.contiguous()  # (d_model, k_r)

    # --- Step 3: project out retain directions ---
    # U_specific = U_f - U_r @ (U_r^T @ U_f)
    overlap = U_r.T @ U_f            # (k_r, k_f): how much each U_f aligns with U_r
    U_specific = U_f - U_r @ overlap  # (d_model, k_f): remove retain components

    # --- Step 4: re-orthonormalise ---
    norms = U_specific.norm(dim=0, keepdim=True).clamp(min=1e-8)
    U_specific = U_specific / norms
    # Keep only columns with enough residual norm (not collapsed by retain projection)
    keep = (norms.squeeze(0) > 0.3)
    if keep.sum() == 0:
        logger.warning("  All forget-specific directions collapsed into retain subspace! "
                       "Falling back to raw forget PCA.")
        return U_f
    U_erase = U_specific[:, keep]

    # QR for exact orthonormality
    Q, _ = torch.linalg.qr(U_erase)
    logger.info(f"  Forget-specific subspace: k_eff={Q.shape[1]} "
                f"(after removing retain overlap)")

    return Q.contiguous()  # (d_model, k_eff)


# ======================================================================
# Weight editor
# ======================================================================

@dataclass
class SPWEConfig:
    """Configuration for Subspace-Projected Weight Editing."""
    k: int = 5               # forget PCA components
    retain_k: int = 10       # retain PCA components to suppress (overlap removal)
    lambda_erase: float = 0.5  # erasure strength in [0, 1]
    edit_down_proj: bool = True   # edit MLP down_proj
    edit_o_proj: bool = True      # edit attention o_proj
    layers: List[int] = field(default_factory=lambda: [19])
    device: str = 'cuda'


class SubspaceWeightEditor:
    """
    Permanently edit LLaMA model weights to erase forget knowledge.

    Usage:
        editor = SubspaceWeightEditor(model, config)
        editor.edit(layer_idx, forget_activations, retain_activations)
        # model weights are now modified in-place
    """

    def __init__(self, model, config: Optional[SPWEConfig] = None):
        self.model  = model
        self.config = config or SPWEConfig()
        self.edit_log: Dict[int, dict] = {}

    def _get_layer(self, layer_idx: int):
        return self.model.model.layers[layer_idx]

    def _project_out(
        self,
        W: torch.Tensor,        # (d_model, *)  weight matrix
        U: torch.Tensor,        # (d_model, k)  subspace to erase
        lam: float,
    ) -> torch.Tensor:
        """
        W_new = W - lambda * U @ (U^T @ W)
              = (I - lambda * U @ U^T) @ W

        Works for any W with first dimension d_model.
        Done in float32 then cast back to W's original dtype.
        """
        orig_dtype = W.dtype
        W_f  = W.float()
        U_f  = U.float().to(W.device)
        # U^T @ W: (k, *)
        UTW  = U_f.T @ W_f          # (k, *)
        # Subtract projection
        W_new = W_f - lam * (U_f @ UTW)
        return W_new.to(orig_dtype)

    def edit(
        self,
        layer_idx: int,
        forget_activations: torch.Tensor,  # (N_f, d_model)
        retain_activations: torch.Tensor,  # (N_r, d_model)
    ):
        """
        Edit weights of layer_idx in-place.

        Args:
            layer_idx:          transformer layer index
            forget_activations: last-token forget activations (N_f, d_model)
            retain_activations: last-token retain activations (N_r, d_model)
        """
        cfg = self.config
        logger.info(f"\n[SPWE] Editing layer {layer_idx} "
                    f"(lambda={cfg.lambda_erase}, k={cfg.k})")

        # --- Compute forget-specific subspace ---
        U_erase = compute_forget_specific_subspace(
            forget_activations=forget_activations,
            retain_activations=retain_activations,
            k=cfg.k,
            retain_k=cfg.retain_k,
        )  # (d_model, k_eff) on CPU

        layer = self._get_layer(layer_idx)
        log = {'k_eff': U_erase.shape[1]}

        # --- Edit MLP down_proj: (d_model, d_ffn) ---
        if cfg.edit_down_proj:
            W = layer.mlp.down_proj.weight.data  # (d_model, d_ffn)
            W_new = self._project_out(W, U_erase, cfg.lambda_erase)
            delta = (W_new - W).norm().item()
            layer.mlp.down_proj.weight.data = W_new
            log['down_proj_delta'] = delta
            logger.info(f"  down_proj edited: ||delta||={delta:.4f}")

        # --- Edit attention o_proj: (d_model, d_model) ---
        if cfg.edit_o_proj:
            W = layer.self_attn.o_proj.weight.data  # (d_model, d_model)
            W_new = self._project_out(W, U_erase, cfg.lambda_erase)
            delta = (W_new - W).norm().item()
            layer.self_attn.o_proj.weight.data = W_new
            log['o_proj_delta'] = delta
            logger.info(f"  o_proj edited:    ||delta||={delta:.4f}")

        self.edit_log[layer_idx] = log
        logger.info(f"[SPWE] Layer {layer_idx} done.")

    def edit_all(
        self,
        forget_activations_by_layer: Dict[int, torch.Tensor],
        retain_activations_by_layer: Dict[int, torch.Tensor],
    ):
        """Edit all configured layers."""
        for layer_idx in self.config.layers:
            self.edit(
                layer_idx=layer_idx,
                forget_activations=forget_activations_by_layer[layer_idx],
                retain_activations=retain_activations_by_layer[layer_idx],
            )

    def summary(self):
        for layer_idx, log in self.edit_log.items():
            logger.info(
                f"Layer {layer_idx}: k_eff={log['k_eff']}, "
                f"down_proj_delta={log.get('down_proj_delta', 'N/A'):.4f}, "
                f"o_proj_delta={log.get('o_proj_delta', 'N/A'):.4f}"
            )
