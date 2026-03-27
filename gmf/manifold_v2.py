# GMF v2 - PCA Subspace Manifold
#
# Key upgrade over v1:
#   v1: ForgetManifold = single Gaussian (mean + diagonal covariance)
#   v2: ForgetSubmanifold = PCA subspace (mean + top-k principal components)
#
# Theoretical grounding:
#   The Linear Representation Hypothesis (Zou et al. 2023, Park et al. 2023)
#   shows that specific knowledge in LLMs resides on low-dimensional LINEAR
#   subspaces of activation space.
#
#   LUNAR uses k=1 (a single direction).
#   GMF v2 uses k>=2 (the full subspace), making it strictly more general.

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm


@dataclass
class ForgetSubmanifold:
    """
    PCA subspace representation of forget knowledge.

    Formal definition:
        M_f = { a in R^d : || Lambda^{-1/2} U^T (a - mu) || <= tau }

    where:
        mu  : (d_model,)   mean of forget activations
        U   : (d_model, k) top-k principal components (orthonormal columns)
        S   : (k,)         singular values = sqrt(eigenvalues), i.e. std per component
        k   : number of components retained

    Distance metric (within-subspace Mahalanobis):
        d(x) = || Lambda^{-1/2} U^T (x - mu) ||
             = || diag(1/S) * U^T * (x - mu) ||

    When k=1, U[:,0] is the LUNAR refusal direction and S[0] is its std.
    """
    mu: torch.Tensor      # (d_model,)
    U: torch.Tensor       # (d_model, k)
    S: torch.Tensor       # (k,)  singular values
    k: int
    epsilon: float = 1e-6

    def subspace_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mahalanobis distance in the k-dimensional PCA subspace.

        d(x) = || diag(1/S) * U^T * (x - mu) ||

        Args:
            x: (batch, d_model) or (d_model,)
        Returns:
            (batch,) distances
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        diff = x - self.mu.unsqueeze(0)          # (batch, d_model)
        proj = diff @ self.U                      # (batch, k)
        normalized = proj / (self.S + self.epsilon).unsqueeze(0)  # (batch, k)
        dist = normalized.norm(dim=-1)            # (batch,)

        return dist.squeeze(0) if squeeze else dist

    def projection_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw projection magnitude onto the subspace (un-normalized).
        Useful for visualization.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        diff = x - self.mu.unsqueeze(0)
        proj = diff @ self.U                      # (batch, k)
        mag = proj.norm(dim=-1)
        return mag.squeeze(0) if squeeze else mag

    def explained_variance_ratio(self) -> torch.Tensor:
        """Fraction of variance explained by each component."""
        var = self.S ** 2
        return var / var.sum()

    def to(self, device):
        self.mu = self.mu.to(device).float()
        self.U = self.U.to(device).float()
        self.S = self.S.to(device).float()
        return self


@dataclass
class AttractorManifold:
    """
    Attractor: target region we push forget activations toward.

    Constructed as: mu_a = mu_f + direction * scale
    where direction is the LUNAR-style refusal direction.
    """
    mu: torch.Tensor        # (d_model,) attractor center
    direction: torch.Tensor # (d_model,) normalized refusal direction
    scale: float = 1.0

    def to(self, device):
        self.mu = self.mu.to(device).float()
        self.direction = self.direction.to(device).float()
        return self


class ManifoldExtractorV2:
    """
    Extract PCA subspace manifold from LLM activations.

    Usage:
        extractor = ManifoldExtractorV2(hidden_size=4096, k=10)
        submanifold = extractor.extract_forget_submanifold(forget_activations)
        attractor   = extractor.extract_attractor(refusal_direction, submanifold)
    """

    def __init__(
        self,
        hidden_size: int,
        k: int = 10,
        epsilon: float = 1e-6,
        device: str = 'cuda',
    ):
        self.hidden_size = hidden_size
        self.k = k
        self.epsilon = epsilon
        self.device = device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_last_token(activations: List[torch.Tensor]) -> torch.Tensor:
        """Stack last-token activations from a list of tensors."""
        all_acts = []
        for act in activations:
            act = act.cpu().float()
            if act.dim() == 3:          # (batch, seq, d)
                all_acts.append(act[:, -1, :])
            elif act.dim() == 2:        # (seq, d)
                all_acts.append(act[-1:, :])
            else:                       # (d,)
                all_acts.append(act.unsqueeze(0))
        return torch.cat(all_acts, dim=0)  # (N, d_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_forget_submanifold(
        self,
        activations: List[torch.Tensor],
    ) -> ForgetSubmanifold:
        """
        Extract PCA subspace from forget activations.

        Steps:
            1. Stack last-token activations -> (N, d_model)
            2. Compute mean mu
            3. SVD on centered data -> top-k principal components U, S

        Args:
            activations: list of per-sample activation tensors

        Returns:
            ForgetSubmanifold
        """
        print("Extracting PCA submanifold...")

        data = self._collect_last_token(activations)   # (N, d_model)
        N, D = data.shape
        print(f"  Samples: {N}, Dimensions: {D}")

        mu = data.mean(dim=0)                          # (d_model,)
        centered = data - mu.unsqueeze(0)              # (N, d_model)

        k = min(self.k, N - 1, D)

        # --- SVD: centered = U_svd @ diag(S_svd) @ Vh_svd ---
        # Principal components = rows of Vh_svd (= columns of V)
        try:
            # torch.linalg.svd returns U(N,N), S(min(N,D)), Vh(D,D) for full
            # Use full_matrices=False for thin SVD
            _, S_svd, Vh_svd = torch.linalg.svd(centered, full_matrices=False)
            # Vh_svd: (min(N,D), D) -> rows are principal directions
            U_pca = Vh_svd[:k, :].T.contiguous()       # (d_model, k)
            # Normalize singular values to per-sample std
            S_pca = S_svd[:k] / (N - 1) ** 0.5        # (k,)
        except Exception as e:
            print(f"  torch SVD failed ({e}), falling back to numpy...")
            _, S_np, Vh_np = np.linalg.svd(centered.numpy(), full_matrices=False)
            U_pca = torch.from_numpy(Vh_np[:k, :].T.copy()).float()
            S_pca = torch.from_numpy(S_np[:k] / (N - 1) ** 0.5).float()

        var_ratio = (S_pca ** 2) / (S_pca ** 2).sum()
        print(f"  Retained k={k} components")
        print(f"  Explained variance (top 3): "
              f"{var_ratio[:3].tolist()}")
        print(f"  Cumulative variance (top k): {var_ratio.sum().item():.3f}")

        return ForgetSubmanifold(
            mu=mu,
            U=U_pca,
            S=S_pca,
            k=k,
            epsilon=self.epsilon,
        )

    def extract_retain_mean(
        self,
        retain_activations: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute mean of retain last-token activations.

        Returns:
            mu_r: (d_model,) float32
        """
        data = self._collect_last_token(retain_activations)  # (N, d_model)
        mu_r = data.mean(dim=0)
        print(f"  Retain mean computed from {data.shape[0]} samples, "
              f"||mu_r||={mu_r.norm():.3f}")
        return mu_r

    def extract_attractor(
        self,
        refusal_direction: torch.Tensor,
        forget_submanifold: ForgetSubmanifold,
        attractor_offset: float = 2.0,
    ) -> AttractorManifold:
        """
        Construct attractor manifold from refusal direction.

        attractor_mu = forget_mu + refusal_direction_normalized * attractor_offset

        Args:
            refusal_direction: LUNAR-style mean-diff direction (d_model,)
            forget_submanifold: the extracted forget submanifold
            attractor_offset: how far along the direction to place attractor

        Returns:
            AttractorManifold
        """
        direction = refusal_direction.float().cpu()
        direction = direction / (direction.norm() + 1e-8)

        mu_a = forget_submanifold.mu + direction * attractor_offset

        print(f"Attractor offset={attractor_offset:.2f}, "
              f"||direction||={direction.norm():.4f}")

        return AttractorManifold(
            mu=mu_a,
            direction=direction,
            scale=attractor_offset,
        )
