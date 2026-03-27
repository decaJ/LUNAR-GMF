# attacks/attack4_softprompt.py
#
# Attack 4: Orthogonal Soft-Prompt Attack (White-box)
# =====================================================
# Optimises a continuous soft-prompt prefix whose last-token activation
# at layer l is maximally orthogonal to the LUNAR steering direction v̂,
# while still eliciting the forget answer (cross-entropy loss on target).
#
# Optimisation objective:
#   min_{P}  CE(P ∥ x_forget → y_forget)  −  λ · cos²(h_l(P ∥ x), v̂)
#
# After optimisation the soft-prompt embeddings are projected onto the
# nearest real tokens (greedy nearest-neighbour in embedding space) so
# they can be prepended to any question using standard token-level generation.
#
# Why it threatens LUNAR:
#   The orthogonality penalty explicitly forces the activation away from v̂.
#   LUNAR's weight edit suppresses knowledge retrieval along v̂; an activation
#   perpendicular to v̂ bypasses this suppression entirely.
#
# Why GMF is more robust:
#   To elicit the forget answer the model must process the forget question,
#   which naturally draws the residual stream close to μ_f (the centroid of
#   the forget manifold).  Even if the soft-prompt shifts the activation off
#   v̂, it cannot simultaneously push the activation far from μ_f — that would
#   break the semantic information needed to produce the answer.
#   The Gaussian gate is distance-based, not direction-based, so it still fires.

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class OrthogonalSoftPromptAttack:
    """White-box orthogonal soft-prompt attack.

    Args:
        model_base:     loaded model wrapper
        direction:      LUNAR steering direction v (d_model,)
        layer_idx:      transformer layer where activation is measured
        n_tokens:       number of learnable prefix tokens
        n_steps:        gradient-descent optimisation steps
        lr:             learning rate for Adam
        lambda_orth:    weight for the orthogonality penalty
    """

    def __init__(
        self,
        model_base,
        direction: Tensor,
        layer_idx: int,
        n_tokens: int = 20,
        n_steps: int = 200,
        lr: float = 5e-3,
        lambda_orth: float = 3.0,
    ):
        self.model_base  = model_base
        self.layer_idx   = layer_idx
        self.n_tokens    = n_tokens
        self.n_steps     = n_steps
        self.lr          = lr
        self.lambda_orth = lambda_orth

        device = next(model_base.model.parameters()).device
        d_model = model_base.model.config.hidden_size

        # Normalise the LUNAR direction
        self.direction = F.normalize(direction.float().to(device), dim=-1)  # (d,)

        # Learnable soft-prompt embeddings (near zero init → near-identity start)
        self.soft_embeds = nn.Parameter(
            torch.randn(1, n_tokens, d_model, device=device, dtype=torch.float32) * 0.01
        )

        self._act_cache: List[Tensor] = []

    # ── Internals ─────────────────────────────────────────────────

    def _hook_fn(self, module, input, output):
        """Forward hook: cache last-token activation at layer_idx."""
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
        # store last-token position, float32
        self._act_cache.append(act[:, -1, :].float())

    def _get_embed_layer(self):
        return self.model_base.model.model.embed_tokens

    # ── Optimisation ──────────────────────────────────────────────

    def optimize(
        self,
        forget_qa: List[Tuple[str, str]],
        n_samples: int = 4,
        verbose: bool = True,
    ) -> Tensor:
        """Run gradient-descent to optimise the soft-prompt embeddings.

        Uses up to n_samples forget QA pairs per update step.

        Returns:
            soft_embeds: (1, n_tokens, d_model) optimised embeddings
        """
        device    = next(self.model_base.model.parameters()).device
        tokenizer = self.model_base.tokenizer
        embed_layer = self._get_embed_layer()
        optimizer = torch.optim.Adam([self.soft_embeds], lr=self.lr)

        # Freeze model weights
        for p in self.model_base.model.parameters():
            p.requires_grad_(False)

        pairs = forget_qa[:n_samples]
        direction = self.direction  # already on device

        for step in range(self.n_steps):
            step_ce = 0.0
            step_orth = 0.0
            optimizer.zero_grad()

            for question, answer in pairs:
                # ── Tokenise question + answer ─────────────────────
                qa_str = f"[INST] {question} [/INST] {answer}"
                tokens = tokenizer(
                    qa_str,
                    return_tensors="pt",
                    add_special_tokens=True,
                ).to(device)
                input_ids = tokens.input_ids  # (1, L)

                # Embeddings of question+answer (no grad)
                with torch.no_grad():
                    qa_embeds = embed_layer(input_ids).float()  # (1, L, d)

                # Prepend soft prompts
                full_embeds = torch.cat(
                    [self.soft_embeds, qa_embeds], dim=1
                )  # (1, n_tokens+L, d)

                # Build labels: -100 for soft-prompt region, token IDs for answer part
                # Find where the answer starts in the token sequence
                answer_tokens = tokenizer(
                    answer,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids.to(device)
                ans_len = answer_tokens.shape[1]

                # label shape: (1, n_tokens + L)
                labels = torch.full(
                    (1, self.n_tokens + input_ids.shape[1]),
                    -100,
                    dtype=torch.long,
                    device=device,
                )
                # Only supervise answer tokens (last ans_len positions of the sequence)
                labels[0, -ans_len:] = input_ids[0, -ans_len:]

                # ── Register activation hook ───────────────────────
                self._act_cache.clear()
                handle = self.model_base.model_block_modules[
                    self.layer_idx
                ].register_forward_hook(self._hook_fn)

                try:
                    outputs = self.model_base.model.model(
                        inputs_embeds=full_embeds
                    )
                    logits = self.model_base.model.lm_head(
                        outputs.last_hidden_state
                    )  # (1, n_tokens+L, vocab)

                    # CE loss (only on answer tokens)
                    shift_logits = logits[0, :-1, :]   # (n_tokens+L-1, vocab)
                    shift_labels = labels[0, 1:]        # (n_tokens+L-1,)
                    ce_loss = F.cross_entropy(
                        shift_logits, shift_labels, ignore_index=-100
                    )

                    # Orthogonality penalty: maximise angle w.r.t. LUNAR direction
                    if self._act_cache:
                        act = self._act_cache[0]   # (1, d)
                        cos_sim = F.cosine_similarity(
                            act, direction.unsqueeze(0)
                        )  # (1,)
                        orth_penalty = cos_sim.pow(2).mean()
                    else:
                        orth_penalty = torch.tensor(0.0, device=device)

                    loss = ce_loss - self.lambda_orth * orth_penalty
                    loss.backward()

                    step_ce   += ce_loss.item()
                    step_orth += orth_penalty.item()

                finally:
                    handle.remove()

            optimizer.step()

            if verbose and (step % 50 == 0 or step == self.n_steps - 1):
                logger.info(
                    f"  SoftPrompt step {step:3d}/{self.n_steps}: "
                    f"CE={step_ce/len(pairs):.4f}  "
                    f"orth={step_orth/len(pairs):.4f}"
                )

        # Restore gradients
        for p in self.model_base.model.parameters():
            p.requires_grad_(True)

        return self.soft_embeds.detach()

    # ── Token conversion ──────────────────────────────────────────

    def embeds_to_token_ids(self, soft_embeds: Optional[Tensor] = None) -> Tensor:
        """Find nearest real tokens to each soft-prompt embedding.

        Args:
            soft_embeds: (1, n_tokens, d_model); uses self.soft_embeds if None

        Returns:
            token_ids: (n_tokens,) LongTensor of nearest token IDs
        """
        if soft_embeds is None:
            soft_embeds = self.soft_embeds

        embed_matrix = self._get_embed_layer().weight.detach().float()
        # soft_embeds: (n_tokens, d)
        prompts = soft_embeds.squeeze(0).float()           # (n, d)
        # Pairwise L2 distance → nearest token
        dists = torch.cdist(prompts, embed_matrix)          # (n, vocab)
        token_ids = dists.argmin(dim=-1)                    # (n,)
        return token_ids

    # ── Attack entry point ────────────────────────────────────────

    def attack(
        self,
        forget_qa: List[Tuple[str, str]],
        optimize_first: bool = True,
        verbose: bool = True,
    ) -> List[Tuple[str, str]]:
        """Optimise soft prompts (if needed) and return attacked QA pairs.

        Each forget question is prefixed with the decoded nearest-token
        representation of the optimised soft-prompt embeddings.

        Args:
            forget_qa:       list of (question, answer) for forget set
            optimize_first:  run optimisation; set False to reuse cached

        Returns:
            List of (attacked_question, ground_truth_answer)
        """
        if optimize_first:
            if verbose:
                logger.info(
                    f"\n[SoftPrompt] Optimising {self.n_tokens}-token prefix "
                    f"for {self.n_steps} steps (λ_orth={self.lambda_orth})..."
                )
            self.optimize(forget_qa, verbose=verbose)

        token_ids = self.embeds_to_token_ids()  # (n_tokens,)
        tokenizer = self.model_base.tokenizer
        prefix_str = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

        if verbose:
            logger.info(f"  Decoded soft-prompt prefix: '{prefix_str[:80]}...'")

        out = []
        for question, answer in forget_qa:
            attacked_q = f"{prefix_str} {question}"
            out.append((attacked_q, answer))
        return out
