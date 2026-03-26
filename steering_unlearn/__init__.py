# Copyright (c) Steering Unlearn Project
"""
Steering Unlearn Package

This package implements Controlled Region-to-Attractor Transport for LLM Unlearning.

Key Components:
1. dynamics: Core dynamics-based unlearning modules
   - recoverability_region: Defines R_f^{(l)} = { a : r^{(l)}(a) >= tau_f }
   - abstention_attractor: Defines A^{(l)} = { a : s^{(l)}(a) >= tau_a }
   - energy_function: E^{(l)}(a) = φ_r(r) + λ_a * φ_a(s) + λ_d * φ_d(a, a_base)
   - transport_controller: u_l(a) = -η_l * ∇_a E^{(l)}(a)

2. preprocessing: Activation extraction and preprocessing utilities
3. training: Training modules (steering modules, losses)
4. evaluation: Evaluation metrics and utilities

Method Name: Controlled Region-to-Attractor Transport for LLM Unlearning

Core Algorithm:
- Recoverability Region: R_f^{(l)} = { a : r^{(l)}(a) >= tau_f }
- Abstention Attractor: A^{(l)} = { a : s^{(l)}(a) >= tau_a }
- Energy Function: E^{(l)}(a) = φ_r(r) + λ_a * φ_a(s) + λ_d * φ_d(a, a_base)
- Control Law: u_l(a) = -η_l * ∇_a E^{(l)}(a)
- Finite-step Transport: a^+ = a + u(a)
"""

from . import dynamics
from . import preprocessing
from . import training
from . import evaluation

__version__ = "2.0.0"
__all__ = ['dynamics', 'preprocessing', 'training', 'evaluation']