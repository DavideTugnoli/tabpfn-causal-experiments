"""
This module provides functions and a wrapper class to assess the quality of
synthetic data.
"""

import pandas as pd
import numpy as np
import itertools
import random
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# The only required external dependency is SynthEval
from syntheval import SynthEval

# Disable matplotlib plotting to prevent SynthEval from saving images
plt.ioff()
matplotlib.rcParams['figure.max_open_warning'] = 0

# --- Section 1: Standalone Metric Functions ---

def calculate_correlation_metrics(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cat_cols: List[str] = None
) -> Dict[str, float]:
    """
    Calculates max and mean correlation distance.

    Args:
        real_data: DataFrame of real data.
        synthetic_data: DataFrame of synthetic data.
        cat_cols: List of categorical column names.

    Returns:
        A dictionary with 'max_corr_difference' and 'mean_corr_difference'.
    """
    if cat_cols is None:
        cat_cols = []

    evaluator = SynthEval(real_data, cat_cols=cat_cols)
    
    # BUGFIX: Temporarily disable plt.savefig to prevent SynthEval from creating plot files
    original_savefig = plt.savefig
    plt.savefig = lambda *args, **kwargs: None
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            evaluator.evaluate(
                synthetic_data,
                None,
                **{"corr_diff": {"return_mats": True}}
            )
    finally:
        # Restore the original savefig function
        plt.savefig = original_savefig

    metrics = {}
    try:
        corr_diff_results = evaluator._raw_results["corr_diff"]
        diff_matrix = corr_diff_results["diff_cor_mat"]
        abs_diff_values = np.abs(diff_matrix.values)
        metrics['max_corr_difference'] = np.max(abs_diff_values)
        upper_triangle_indices = np.triu_indices(abs_diff_values.shape[0], k=1)
        metrics['mean_corr_difference'] = np.mean(abs_diff_values[upper_triangle_indices])
    except Exception:
        metrics['max_corr_difference'] = -1.0
        metrics['mean_corr_difference'] = -1.0
    return metrics

def calculate_propensity_metrics(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cat_cols: List[str] = None
) -> Dict[str, float]:
    """
    Calculates the Propensity Mean Squared Error (pMSE) and related metrics.

    Args:
        real_data: DataFrame of real data.
        synthetic_data: DataFrame of synthetic data.
        cat_cols: List of categorical column names.

    Returns:
        Dictionary with the following keys:
            - 'avg pMSE'
            - 'pMSE err'
            - 'avg acc'
            - 'acc err'
        If a metric is missing, its value is -1.0.
    """
    if cat_cols is None:
        cat_cols = []

    evaluator = SynthEval(real_data, cat_cols=cat_cols)
    
    # BUGFIX: Temporarily disable plt.savefig to prevent SynthEval from creating plot files
    original_savefig = plt.savefig
    plt.savefig = lambda *args, **kwargs: None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            evaluator.evaluate(
                synthetic_data,
                None,
                **{"p_mse": {"k_folds": 5, "max_iter": 100, "solver": "liblinear"}}
            )
    finally:
        # Restore the original savefig function
        plt.savefig = original_savefig

    keys = ['avg pMSE', 'pMSE err', 'avg acc', 'acc err']
    try:
        pmse_results = evaluator._raw_results["p_mse"]
        return {key: pmse_results.get(key, -1.0) for key in keys}
    except Exception:
        return {key: -1.0 for key in keys}

def _discretize_for_kmarginal(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Internal function to replicate the percentile-rank-based binning logic.
    Categorical columns are passed through without modification.
    Numerical columns with more than 20 unique values are discretized into 20 bins
    using percentile rank binning.
    """
    numeric_features = [
        col for col in real_data.columns
        if col not in cat_cols and real_data[col].nunique() >= 20
    ]

    real_binned = real_data.copy()
    for col in numeric_features:
        not_na_mask = real_binned[col].notna()
        ranked_pct = real_binned.loc[not_na_mask, col].rank(pct=True)
        real_binned.loc[not_na_mask, col] = ranked_pct.apply(lambda x: int(20 * x) if x < 1 else 19)
        real_binned.loc[real_data[col].isna(), col] = -1

    syn_binned = synthetic_data.copy()
    for col in numeric_features:
        syn_not_na_mask = syn_binned[col].notna()
        syn_numeric_values = pd.to_numeric(syn_binned.loc[syn_not_na_mask, col])
        binned_syn_values = syn_numeric_values.copy()
        max_value_of_previous_bin = -np.inf
        unique_bins = sorted([b for b in real_binned[col].unique() if b != -1])

        for i, bin_val in enumerate(unique_bins):
            indices_for_bin = real_binned[real_binned[col] == bin_val].index
            max_value_in_bin = real_data.loc[indices_for_bin, col].max()

            min_value_of_current_bin = max_value_of_previous_bin
            condition = (syn_numeric_values > min_value_of_current_bin)
            if i < len(unique_bins) - 1:
                condition &= (syn_numeric_values <= max_value_in_bin)

            binned_syn_values.loc[condition] = bin_val
            max_value_of_previous_bin = max_value_in_bin

        syn_binned.loc[syn_not_na_mask, col] = binned_syn_values
        syn_binned.loc[synthetic_data[col].isna(), col] = -1

    # Ensure all columns (numeric and categorical) are returned
    all_cols = numeric_features + cat_cols
    return real_binned[all_cols].astype(int), syn_binned[all_cols].astype(int)

def calculate_kmarginal_tvd(
    
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cat_cols: List[str] = None,
    k: int = 2
) -> float:
    """
    Calculates the K-Marginal Total Variation Distance (TVD).

    Args:
        real_data: DataFrame of real data.
        synthetic_data: DataFrame of synthetic data.
        cat_cols: List of categorical column names.
        k: The order of marginals to compute.

    Returns:
        The mean TVD score.
    """
    if cat_cols is None:
        cat_cols = []
    
    real_processed, syn_processed = _discretize_for_kmarginal(real_data, synthetic_data, cat_cols)
    features = real_processed.columns.tolist()

    if len(features) < k: return 1.0

    marginals = list(itertools.combinations(sorted(features), k))
    if len(marginals) > 1000:
        marginals = random.sample(marginals, 1000)

    if not marginals: return 1.0

    total_density_diff_sum = 0
    for marg in marginals:
        marg = list(marg)
        t_den = real_processed.groupby(marg).size() / len(real_processed)
        s_den = syn_processed.groupby(marg).size() / len(syn_processed)
        abs_den_diff = t_den.subtract(s_den, fill_value=0).abs()
        total_density_diff_sum += abs_den_diff.sum()

    mean_total_density_diff = total_density_diff_sum / len(marginals)
    mean_tvd = mean_total_density_diff / 2.0
    return mean_tvd

# --- Section 2: Wrapper Class for Usability ---

class FaithfulDataEvaluator:
    """
    A wrapper class that provides a single entry point to calculate all
    faithful metrics, maintaining usability for experiments.
    """
    def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        categorical_columns: List[str] = None,
        k_for_kmarginal: int = 2
    ) -> Dict[str, float]:
        """
        Runs the complete evaluation by calling the standalone metric functions.

        Args:
            real_data: The DataFrame of real data.
            synthetic_data: The DataFrame of synthetic data.
            categorical_columns: A list of column names that are categorical.
            k_for_kmarginal: The order of marginals for the k-marginal metric.

        Returns:
            A dictionary containing all calculated metric scores.
        """
        if categorical_columns is None:
            categorical_columns = []

        results = {}

        # Call each standalone function
        corr_metrics = calculate_correlation_metrics(real_data, synthetic_data, categorical_columns)
        results.update(corr_metrics)

        results['propensity_metrics'] = calculate_propensity_metrics(real_data, synthetic_data, categorical_columns)

        results['k_marginal_tvd'] = calculate_kmarginal_tvd(real_data, synthetic_data, categorical_columns, k=k_for_kmarginal)

        return results

# --- Section 3: Example of How to Use This File ---
if __name__ == '__main__':
    print("Example usage of the modular metric functions and wrapper class")

    # 1. Create sample data with a categorical feature
    np.random.seed(42)
    sample_real_data = pd.DataFrame({
        'numeric_A': np.random.randn(500),
        'numeric_B': np.random.rand(500) * 100,
        'category_C': np.random.choice(['A', 'B', 'C'], 500, p=[0.5, 0.3, 0.2])
    })
    # For SynthEval, categorical columns need to be integer-encoded
    sample_real_data['category_C_encoded'] = sample_real_data['category_C'].astype('category').cat.codes

    sample_synthetic_data = pd.DataFrame({
        'numeric_A': np.random.randn(500) * 1.2,
        'numeric_B': np.random.rand(500) * 90,
        'category_C': np.random.choice(['A', 'B', 'C'], 500, p=[0.45, 0.35, 0.2])
    })
    sample_synthetic_data['category_C_encoded'] = sample_synthetic_data['category_C'].astype('category').cat.codes
    
    # Define which columns are categorical for the evaluators
    # Note: We use the encoded column for SynthEval, but the original for our k-marginal logic
    cat_cols_for_eval = ['category_C_encoded']

    # --- USAGE 1: Calling functions individually ---
    print("\n--- 1. Calling functions individually ---")
    corr_res = calculate_correlation_metrics(sample_real_data, sample_synthetic_data, cat_cols_for_eval)
    print(f"Correlation metrics: {corr_res}")

    pmse_res = calculate_propensity_metrics(sample_real_data, sample_synthetic_data, cat_cols_for_eval)
    print(f"Propensity metrics: {pmse_res}")

    # For k-marginal, we pass the original categorical column name
    km_res = calculate_kmarginal_tvd(
        sample_real_data[['numeric_A', 'numeric_B', 'category_C_encoded']],
        sample_synthetic_data[['numeric_A', 'numeric_B', 'category_C_encoded']],
        cat_cols=['category_C_encoded'],
        k=2
    )
    print(f"K-Marginal TVD: {km_res:.6f}")

    # --- USAGE 2: Using the convenient wrapper class ---
    print("\n--- 2. Using the wrapper class for all metrics ---")
    evaluator = FaithfulDataEvaluator()
    all_metrics = evaluator.evaluate(
        real_data=sample_real_data,
        synthetic_data=sample_synthetic_data,
        categorical_columns=cat_cols_for_eval,
        k_for_kmarginal=2
    )
    
    for metric_name, value in all_metrics.items():
        print(f"{metric_name:<25}: {value:.6f}")
    print("---------------------------------------------")