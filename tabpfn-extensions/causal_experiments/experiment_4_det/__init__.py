"""
Experiment 4: Causal Knowledge Level Impact on TabPFN Performance.

This module tests how different levels of causal knowledge, derived from
a discovered CPDAG using causal discovery algorithms, affect TabPFN's 
synthetic data generation performance.
"""

from .experiment_4 import run_experiment_4

__all__ = ['run_experiment_4'] 