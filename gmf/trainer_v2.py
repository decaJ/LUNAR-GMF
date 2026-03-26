# GMF v2 - Trainer
#
# Much simpler than v1 trainer:
#   v1: trains an MLP (thousands of weights) with complex multi-objective loss
#   v2: optionally trains 1-2 scalars (step_size, sigma) with simple MSE loss
#
# Two modes:
#   Mode A (no_train=True):
#       No training. step_size is fixed. Use this for quick validation.
#       Equivalent to a closed-form solution: move forget activations
#       fully toward the attractor.
#
#   Mode B (no_train=False):
#       Train step_size (and optionally sigma) with:
#           L = lambda_f * MSE(transported_forget, attractor_mu)
#             + lambda_r * MSE(transported_retain, retain_input)
#       Very fast: converges in a few epochs with tiny overhead.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
import os
import json
import logging

from .manifold_v2 import (
    ForgetSubmanifold,
    AttractorManifold,
    ManifoldExtractorV2,
)
from .module_v2 import GatedODEFlow

logger = logging.getLogger(__name__)


# ======================================================================
# Config
# ======================================================================

@dataclass
class GMFV2Config:
    """Configuration for GMF v2 trainer."""

    # ---- Manifold extraction ----
    pca_k: int = 10                # number of PCA components
    attractor_offset: float = 2.0  # distance from forget mean to attractor

    # ---- Gate ----
    sigma: float = 1.0
    learnable_sigma: bool = False

    # ---- ODE transport ----
    num_ode_steps: int = 5
    step_size: float = 1.0         # initial step size
    learnable_step: bool = True    # train step_size?

    # ---- Training (Mode B only) ----
    no_train: bool = False         # True = skip training entirely (Mode A)
    num_epochs: int = 10           # fewer epochs needed vs v1
    batch_size: int = 32
    learning_rate: float = 1e-2    # larger LR is fine for 1-2 scalars
    weight_decay: float = 0.0

    # ---- Loss weights ----
    lambda_forget: float = 1.0     # pull forget activations to attractor
    lambda_retain: float = 1.0     # keep retain activations unchanged

    # ---- Misc ----
    device: str = 'cuda'
    save_dir: str = 'checkpoints/gmf_v2'
    log_every: int = 5


# ======================================================================
# Dataset
# ======================================================================

class ActivationDataset(Dataset):
    def __init__(self, inputs: torch.Tensor):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


# ======================================================================
# Trainer
# ======================================================================

class GMFV2Trainer:
    """
    Two-phase trainer for GMF v2.

    Phase 1 (offline): extract PCA submanifold + attractor from activations
    Phase 2 (optional): train step_size scalar with simple MSE losses
    """

    def __init__(
        self,
        hidden_size: int,
        config: Optional[GMFV2Config] = None,
        layer_idx: int = 0,
    ):
        self.hidden_size = hidden_size
        self.config      = config or GMFV2Config()
        self.device      = self.config.device
        self.layer_idx   = layer_idx

        # Built in phase 1
        self.forget_submanifold: Optional[ForgetSubmanifold] = None
        self.attractor: Optional[AttractorManifold] = None

        # Built in phase 2
        self.module: Optional[GatedODEFlow] = None

        self.history = {
            'total_loss': [],
            'forget_loss': [],
            'retain_loss': [],
            'step_size': [],
        }

    # ------------------------------------------------------------------
    # Phase 1: manifold extraction
    # ------------------------------------------------------------------

    def phase1_extract(
        self,
        forget_activations: List[torch.Tensor],
        retain_activations: List[torch.Tensor],
        refusal_direction: torch.Tensor,
    ) -> Tuple[ForgetSubmanifold, AttractorManifold]:
        """
        Extract PCA submanifold and attractor from data.

        Args:
            forget_activations : list of per-sample activation tensors
            retain_activations : list of per-sample activation tensors (unused here)
            refusal_direction  : (d_model,) LUNAR-style direction

        Returns:
            (ForgetSubmanifold, AttractorManifold)
        """
        logger.info(f"[Layer {self.layer_idx}] Phase 1: Extracting PCA submanifold...")

        extractor = ManifoldExtractorV2(
            hidden_size=self.hidden_size,
            k=self.config.pca_k,
            device=self.device,
        )

        self.forget_submanifold = extractor.extract_forget_submanifold(
            forget_activations
        )
        self.attractor = extractor.extract_attractor(
            refusal_direction=refusal_direction,
            forget_submanifold=self.forget_submanifold,
            attractor_offset=self.config.attractor_offset,
        )

        # Move to device
        self.forget_submanifold = self.forget_submanifold.to(self.device)
        self.attractor          = self.attractor.to(self.device)

        logger.info(
            f"[Layer {self.layer_idx}] "
            f"Submanifold: k={self.forget_submanifold.k}, "
            f"var_ratio_top3={self.forget_submanifold.explained_variance_ratio()[:3].tolist()}"
        )

        return self.forget_submanifold, self.attractor

    # ------------------------------------------------------------------
    # Phase 2: optional training
    # ------------------------------------------------------------------

    def phase2_train(
        self,
        forget_inputs: torch.Tensor,
        retain_inputs: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Optionally train step_size (and sigma) using simple MSE losses.

        If config.no_train=True, skips training and uses default step_size.

        Loss:
            L = lambda_f * MSE(transport(forget), attractor_mu)
              + lambda_r * MSE(transport(retain), retain)

        The retain loss is naturally small because gate(retain) ~ 0,
        so transport(retain) ~ retain. We include it for robustness.
        """
        # Build the module
        self.module = GatedODEFlow(
            hidden_size=self.hidden_size,
            sigma=self.config.sigma,
            learnable_sigma=self.config.learnable_sigma,
            num_ode_steps=self.config.num_ode_steps,
            step_size=self.config.step_size,
            learnable_step=self.config.learnable_step,
        ).to(self.device)

        self.module.set_manifold(self.forget_submanifold)
        self.module.set_attractor(self.attractor)

        # Mode A: no training
        if self.config.no_train:
            logger.info(
                f"[Layer {self.layer_idx}] Mode A: no training. "
                f"step_size={self.config.step_size:.3f}"
            )
            return self.history

        # Mode B: train scalars
        logger.info(
            f"[Layer {self.layer_idx}] Phase 2: Training step_size scalar..."
        )

        forget_inputs = forget_inputs.to(self.device).float()
        retain_inputs = retain_inputs.to(self.device).float()
        attractor_mu  = self.attractor.mu.to(self.device).float()

        # Only optimise learnable parameters (step_size and/or sigma)
        trainable = [p for p in self.module.parameters() if p.requires_grad]
        if not trainable:
            logger.info(
                f"[Layer {self.layer_idx}] No learnable parameters. Skipping."
            )
            return self.history

        optimizer = optim.Adam(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        forget_loader = DataLoader(
            ActivationDataset(forget_inputs),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        retain_loader = DataLoader(
            ActivationDataset(retain_inputs),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        self.module.train()

        for epoch in range(self.config.num_epochs):
            f_iter = iter(forget_loader)
            r_iter = iter(retain_loader)
            n_batches = max(len(forget_loader), len(retain_loader))

            ep_total, ep_f, ep_r = [], [], []

            for _ in range(n_batches):
                # --- forget batch ---
                try:
                    f_batch = next(f_iter)
                except StopIteration:
                    f_iter = iter(forget_loader)
                    f_batch = next(f_iter)

                # --- retain batch ---
                try:
                    r_batch = next(r_iter)
                except StopIteration:
                    r_iter = iter(retain_loader)
                    r_batch = next(r_iter)

                optimizer.zero_grad()

                # Forget: transport should push to attractor
                f_transported, _ = self.module(f_batch)
                f_loss = nn.functional.mse_loss(
                    f_transported,
                    attractor_mu.unsqueeze(0).expand_as(f_transported),
                )

                # Retain: transport should leave unchanged
                r_transported, _ = self.module(r_batch)
                r_loss = nn.functional.mse_loss(r_transported, r_batch)

                loss = (
                    self.config.lambda_forget * f_loss
                    + self.config.lambda_retain * r_loss
                )

                loss.backward()

                # Clip gradient for the single scalar
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

                ep_total.append(loss.item())
                ep_f.append(f_loss.item())
                ep_r.append(r_loss.item())

            avg_total = sum(ep_total) / len(ep_total)
            avg_f     = sum(ep_f)     / len(ep_f)
            avg_r     = sum(ep_r)     / len(ep_r)
            s         = self.module.transport.step_size.item()

            self.history['total_loss'].append(avg_total)
            self.history['forget_loss'].append(avg_f)
            self.history['retain_loss'].append(avg_r)
            self.history['step_size'].append(s)

            if (epoch + 1) % self.config.log_every == 0 or epoch == 0:
                logger.info(
                    f"[Layer {self.layer_idx}] Epoch {epoch+1:3d}: "
                    f"total={avg_total:.4f}, "
                    f"forget={avg_f:.4f}, "
                    f"retain={avg_r:.4f}, "
                    f"step_size={s:.4f}"
                )

        self.module.eval()
        logger.info(
            f"[Layer {self.layer_idx}] Training done. "
            f"Final step_size={self.module.transport.step_size.item():.4f}"
        )

        return self.history

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_module(self) -> GatedODEFlow:
        if self.module is None:
            raise RuntimeError("Call phase2_train() first.")
        return self.module

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'module_state': self.module.state_dict(),
            'history':      self.history,
            'config':       self.config,
            'layer_idx':    self.layer_idx,
        }, path)
        logger.info(f"Saved v2 checkpoint to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if self.module is None:
            raise RuntimeError("Build module via phase2_train() before loading.")
        self.module.load_state_dict(ckpt['module_state'])
        self.history = ckpt['history']
        logger.info(f"Loaded v2 checkpoint from {path}")
