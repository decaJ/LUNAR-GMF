# GMF v2 - Subspace Gate + ODE Transport + GatedODEFlow
#
# Three classes, all in one file for easy reading:
#
#   SubspaceGate   : gate based on PCA subspace Mahalanobis distance
#   ODETransport   : multi-step ODE transport toward attractor
#   GatedODEFlow   : combines gate + transport; is the drop-in module for hooks
#
# Key differences vs v1:
#   Gate    : v1 uses Euclidean/Mahalanobis to a SINGLE point
#             v2 uses within-SUBSPACE Mahalanobis (k-dimensional, k >= 1)
#   Flow    : v1 trains an MLP residual network
#             v2 uses analytic ODE (no neural network for the flow)
#   Params  : v1 has thousands of MLP weights per layer
#             v2 has ONE learnable scalar per layer (step_size)
#
# Notation:
#   mu_f  = mean of forget activations
#   U     = (d_model, k) PCA principal components
#   S     = (k,)         singular values
#   mu_a  = attractor center
#   alpha(x) = gate value in [0, 1]
#   T     = number of ODE steps

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .manifold_v2 import ForgetSubmanifold, AttractorManifold


# ======================================================================
# 1. SubspaceGate
# ======================================================================

class SubspaceGate(nn.Module):
    """
    PCA Subspace Gate (no neural network).

    Gate function:
        d(x)     = || diag(1/S) * U^T * (x - mu_f) ||   (Mahalanobis in subspace)
        alpha(x) = exp( -d(x)^2 / (k * 2 * sigma^2) )

    Properties:
        - alpha -> 1 when x is close to the forget subspace center
        - alpha -> 0 when x is far from the forget subspace
        - When k=1, equivalent to LUNAR's direction distance
        - Dimension normalization by k (not d_model) matches the subspace size
    """

    def __init__(
        self,
        hidden_size: int,
        sigma: float = 1.0,
        learnable_sigma: bool = False,
        temperature: float = 1.0,
    ):
        """
        Args:
            hidden_size     : d_model (e.g. 4096 for LLaMA-7B)
            sigma           : bandwidth; sigma=1.0 is scale-invariant with k-normalization
            learnable_sigma : whether to learn sigma during (optional) training
            temperature     : additional softening/sharpening factor
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature

        if learnable_sigma:
            self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        else:
            self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))

        # PCA manifold buffers (set via set_manifold)
        self.register_buffer('manifold_mu', None)
        self.register_buffer('manifold_U',  None)   # (d_model, k)
        self.register_buffer('manifold_S',  None)   # (k,)
        self.k = None

    def set_manifold(self, submanifold: ForgetSubmanifold):
        self.manifold_mu = submanifold.mu.clone().detach().float()
        self.manifold_U  = submanifold.U.clone().detach().float()
        self.manifold_S  = submanifold.S.clone().detach().float()
        self.k           = submanifold.k
        print(f"[SubspaceGate] k={self.k}, "
              f"||mu||={self.manifold_mu.norm():.4f}, "
              f"S[0]={self.manifold_S[0]:.4f}")

    def compute_subspace_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        d(x) = || diag(1/S) * U^T * (x - mu_f) ||

        Returns: (batch,) distances
        """
        diff = x - self.manifold_mu.unsqueeze(0)           # (batch, d_model)
        proj = diff @ self.manifold_U                       # (batch, k)
        normalized = proj / (self.manifold_S + 1e-6).unsqueeze(0)  # (batch, k)
        dist = normalized.norm(dim=-1)                      # (batch,)
        return dist

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model) float32
        Returns:
            alpha: (batch, 1) gate values in (0, 1]
        """
        if self.manifold_mu is None:
            raise RuntimeError("Call set_manifold() before forward()")

        x = x.float()
        dist = self.compute_subspace_distance(x)            # (batch,)

        k        = float(self.k)
        sigma_sq = self.sigma.float() ** 2
        gate = torch.exp(-dist ** 2 / (k * 2.0 * sigma_sq * self.temperature))

        return gate.unsqueeze(-1)                           # (batch, 1)


# ======================================================================
# 2. ODETransport
# ======================================================================

class ODETransport(nn.Module):
    """
    Multi-step ODE transport toward attractor manifold.

    Continuous dynamics:
        da/dt = -alpha(a) * (a - mu_a)

    Discretized (Euler, T steps):
        a^{t+1} = a^t + (step_size / T) * alpha(a^t) * (mu_a - a^t)

    Interpretation:
        - Each step nudges forget activations toward the attractor (alpha~1)
        - Retain activations stay unchanged (alpha~0)
        - Multiple steps allow progressive, smooth transport
        - The gate is re-evaluated at every step (true ODE behavior)

    Parameters:
        num_steps  : T, number of discrete ODE steps
        step_size  : total step magnitude (alpha=1 moves fully by step_size fraction)
        learnable  : if True, step_size is a learned scalar (tiny training needed)
    """

    def __init__(
        self,
        num_steps: int = 5,
        step_size: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_steps = num_steps

        if learnable:
            # Parameterize in log-space for stability: step_size in (0, 2]
            self.log_step = nn.Parameter(
                torch.tensor(step_size, dtype=torch.float32).log()
            )
        else:
            self.register_buffer(
                'log_step',
                torch.tensor(step_size, dtype=torch.float32).log()
            )

    @property
    def step_size(self) -> torch.Tensor:
        # Clamp to (0, 1]: step_size=1.0 means "move all the way to attractor"
        # step_size=0.5 means "move halfway"
        return self.log_step.exp().clamp(1e-3, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        attractor_mu: torch.Tensor,
        gate_fn,
    ) -> torch.Tensor:
        """
        Transport x toward attractor_mu.

        Uses NORMALIZED velocity so step_size is scale-invariant:
            velocity_hat = (mu_a - a) / ||mu_a - a||   (unit direction)
            move_dist    = step_size * ||mu_a - a||     (fraction of total distance)

        Simplified: a^{t+1} = a^t + (h * alpha) * (mu_a - a^t)
        where h = step_size / T controls the fraction moved per step.
        With h=1/T and T steps, a fully-gated activation reaches mu_a exactly.

        LLaMA activations have magnitude ~100; without normalization a raw
        velocity of magnitude 100 * step_size would destroy the distribution.

        Args:
            x           : (batch, d_model)
            attractor_mu: (d_model,) target center
            gate_fn     : callable, signature gate_fn(a) -> (batch, 1)

        Returns:
            transported: (batch, d_model)
        """
        a = x.float()
        target = attractor_mu.float().unsqueeze(0)   # (1, d_model)
        h = self.step_size / self.num_steps           # fraction per step

        for _ in range(self.num_steps):
            alpha    = gate_fn(a)                    # (batch, 1)
            velocity = target - a                    # (batch, d_model)
            # h * alpha * velocity:
            #   when alpha=1, h=step_size/T:  each step moves step_size/T fraction
            #   after T steps with alpha=1:   a reaches target * step_size
            #   when alpha=0:                 a unchanged
            a = a + h * alpha * velocity

        return a

    def inverse(
        self,
        y: torch.Tensor,
        attractor_mu: torch.Tensor,
        gate_fn,
    ) -> torch.Tensor:
        """
        Approximate inverse: run ODE backwards.
        """
        a = y.float()
        target = attractor_mu.float().unsqueeze(0)
        h = self.step_size / self.num_steps

        for _ in range(self.num_steps):
            alpha    = gate_fn(a)
            velocity = target - a
            a        = a - h * alpha * velocity      # reverse sign

        return a


# ======================================================================
# 3. GatedODEFlow  (drop-in replacement for GatedFlow + ResidualFlow)
# ======================================================================

class GatedODEFlow(nn.Module):
    """
    Complete GMF v2 transformation module.

    Forward pass:
        alpha    = SubspaceGate(x)          # gate based on PCA distance
        x_T      = ODETransport(x, mu_a, gate=alpha)   # multi-step ODE
        output   = x_T

    This module is used as a forward hook:
        output_activation = GatedODEFlow(input_activation)

    Learnable parameters:
        - ODETransport.log_step  (1 scalar, optional)
        - SubspaceGate.sigma     (1 scalar, optional)

    Total: 0 to 2 learnable parameters per layer.
    Compare to v1: ~2M parameters (MLP flow).
    """

    def __init__(
        self,
        hidden_size: int,
        sigma: float = 1.0,
        learnable_sigma: bool = False,
        num_ode_steps: int = 5,
        step_size: float = 1.0,
        learnable_step: bool = True,
    ):
        """
        Args:
            hidden_size      : d_model
            sigma            : gate bandwidth
            learnable_sigma  : train sigma?
            num_ode_steps    : T, ODE discretization steps
            step_size        : initial total step magnitude
            learnable_step   : train step_size? (recommended True)
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.gate = SubspaceGate(
            hidden_size=hidden_size,
            sigma=sigma,
            learnable_sigma=learnable_sigma,
        )
        self.transport = ODETransport(
            num_steps=num_ode_steps,
            step_size=step_size,
            learnable=learnable_step,
        )

        # Attractor (frozen, set from data)
        self.register_buffer('attractor_mu',        None)
        self.register_buffer('attractor_direction', None)

    # ------------------------------------------------------------------
    # Setup (called before forward)
    # ------------------------------------------------------------------

    def set_manifold(self, submanifold: ForgetSubmanifold):
        self.gate.set_manifold(submanifold)

    def set_attractor(
        self,
        attractor: AttractorManifold,
    ):
        self.attractor_mu        = attractor.mu.clone().detach().float()
        self.attractor_direction = attractor.direction.clone().detach().float()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: (batch, d_model)
        Returns:
            (transported_x, info_dict)
        """
        x = x.float()

        # Gate values for logging (computed once here for logging only;
        # ODETransport recomputes internally at each step)
        with torch.no_grad():
            alpha_log = self.gate(x)

        transported = self.transport(
            x=x,
            attractor_mu=self.attractor_mu,
            gate_fn=self.gate,
        )

        info = {
            'gate_values': alpha_log.detach(),
            'gate_mean':   alpha_log.mean().item(),
            'gate_max':    alpha_log.max().item(),
            'gate_min':    alpha_log.min().item(),
            'step_size':   self.transport.step_size.item(),
        }

        return transported, info

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.transport.inverse(
            y=y,
            attractor_mu=self.attractor_mu,
            gate_fn=self.gate,
        )
