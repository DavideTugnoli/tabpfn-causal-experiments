"""
Experiment 2: Column ordering effects on vanilla TabPFN.

This experiment tests whether column ordering affects synthetic data quality
when TabPFN uses its implicit autoregressive mechanism (no DAG provided).

Usage:
    python experiment_2.py                    # Run full experiment
    python experiment_2.py --no-resume       # Start fresh
"""

import sys
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO
import warnings
import argparse
import random
import hashlib

# Add the causal_experiments directory to the path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# TabPFN imports - use local imports to avoid HPO dependency issues
from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised.unsupervised import TabPFNUnsupervisedModel

# Create a namespace for the unsupervised module
class UnsupervisedNamespace:
    TabPFNUnsupervisedModel = TabPFNUnsupervisedModel

unsupervised = UnsupervisedNamespace()

warnings.filterwarnings('ignore')

# Local imports
from utils.scm_data import generate_scm_data, get_dag_and_config
from utils.metrics import FaithfulDataEvaluator
from utils.dag_utils import get_ordering_strategies, print_dag_info
from utils.checkpoint_utils import save_checkpoint, get_checkpoint_info, cleanup_checkpoint
from utils.experiment_utils import generate_synthetic_data_quiet, reorder_data_and_columns

# Centralized default config
DEFAULT_CONFIG = {
    'train_sizes': [20, 50, 100, 200, 500],
    'ordering_strategies': ['original', 'topological', 'worst', 'random'],
    'n_repetitions': 10,
    'test_size': 2000,
    'n_permutations': 3,
    'metrics': ['mean_corr_difference', 'max_corr_difference', 'propensity_metrics', 'k_marginal_tvd'],
    'include_categorical': False,
    'n_estimators': 3,
    'random_seed_base': 42
}

# Utility: Evaluate metrics

def evaluate_metrics(X_test, X_synth, col_names, categorical_cols, k_for_kmarginal=2):
    evaluator = FaithfulDataEvaluator()
    cat_col_names = [col_names[i] for i in categorical_cols] if categorical_cols else []
    return evaluator.evaluate(
        pd.DataFrame(X_test, columns=col_names),
        pd.DataFrame(X_synth, columns=col_names),
        categorical_columns=cat_col_names if cat_col_names else None,
        k_for_kmarginal=k_for_kmarginal
    )

# Pipeline: Vanilla TabPFN with column reordering

def run_vanilla_tabpfn(X_train, X_test, col_names, categorical_cols, column_order, order_strategy, config, seed, train_size, repetition):
    X_train_reordered, col_names_reordered, categorical_cols_reordered = reorder_data_and_columns(
        X_train, col_names, categorical_cols, column_order
    )
    X_test_reordered, _, _ = reorder_data_and_columns(
        X_test, col_names, categorical_cols, column_order
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    if categorical_cols_reordered:
        model.set_categorical_features(categorical_cols_reordered)
    model.fit(torch.from_numpy(X_train_reordered).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], None, config['n_permutations']
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = evaluate_metrics(X_test_reordered, X_synth, col_names_reordered, categorical_cols_reordered)
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': order_strategy,
        'column_order': str(column_order),
        'graph_type': 'vanilla',
        'graph_structure': '',
    }
    def flatten_metrics():
        flat = {}
        for metric in config['metrics']:
            value = metrics.get(metric)
            if isinstance(value, dict):
                for submetric, subvalue in value.items():
                    flat[f'{metric}_{submetric}'] = subvalue
            else:
                flat[metric] = value
        return flat
    return {**base_info, **flatten_metrics()}

# Main configuration orchestrator

SAVE_DATA_SAMPLES = True  # Set to True to save data_samples for debugging

def hash_array(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()

def run_single_configuration(train_size, order_strategy, repetition, config, 
                           X_test, correct_dag, col_names, categorical_cols, pre_calculated_column_order,
                           data_samples_dir=None, hash_check_dict=None):
    print(f"    Order: {order_strategy}, Rep: {repetition+1}/{config['n_repetitions']}")
    seed = config['random_seed_base'] + repetition
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set PyTorch to deterministic mode for reproducibility
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # For older PyTorch versions
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    X_train_original = generate_scm_data(train_size, seed, config['include_categorical'])
    train_hash = hash_array(X_train_original)
    test_hash = hash_array(X_test)
    if hash_check_dict is not None:
        key = (train_size, repetition)
        if key in hash_check_dict:
            prev_train_hash, prev_test_hash = hash_check_dict[key]
            if prev_train_hash != train_hash or prev_test_hash != test_hash:
                raise RuntimeError(f"[HASH ERROR] Train/Test data hash mismatch for train_size={train_size}, repetition={repetition}!\nPrev train hash: {prev_train_hash}\nCurrent train hash: {train_hash}\nPrev test hash: {prev_test_hash}\nCurrent test hash: {test_hash}")
        else:
            hash_check_dict[key] = (train_hash, test_hash)
    result = run_vanilla_tabpfn(X_train_original, X_test, col_names, categorical_cols, pre_calculated_column_order, order_strategy, config, seed, train_size, repetition)
    # Save data samples if requested
    if SAVE_DATA_SAMPLES and data_samples_dir:
        X_train_reordered, col_names_reordered, _ = reorder_data_and_columns(X_train_original, col_names, categorical_cols, pre_calculated_column_order)
        X_test_reordered, _, _ = reorder_data_and_columns(X_test, col_names, categorical_cols, pre_calculated_column_order)
        file_prefix = f"order_{order_strategy}_size{train_size}_rep{repetition}"
        pd.DataFrame(X_train_reordered, columns=col_names_reordered).head(10).to_csv(data_samples_dir / f"{file_prefix}_train.csv", index=False)
        pd.DataFrame(X_test_reordered, columns=col_names_reordered).head(10).to_csv(data_samples_dir / f"{file_prefix}_test.csv", index=False)
        # For synth, use col_names_reordered
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
        reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
        model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
        if categorical_cols:
            model.set_categorical_features(categorical_cols)
        model.fit(torch.from_numpy(X_train_reordered).float())
        X_synth = generate_synthetic_data_quiet(model, config['test_size'], None, config['n_permutations'])
        pd.DataFrame(X_synth, columns=col_names_reordered).head(10).to_csv(data_samples_dir / f"{file_prefix}_synth.csv", index=False)
    return result

def run_experiment_2(config=None, output_dir="experiment_2_results", resume=True):
    """
    Main experiment function for testing column ordering effects.
    """
    base_config = DEFAULT_CONFIG.copy()
    if config is not None:
        base_config.update(config)
    config = base_config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir) / 'results'
    output_dir.mkdir(exist_ok=True)
    data_samples_dir = Path(script_dir) / 'data_samples'
    data_samples_dir.mkdir(exist_ok=True)
    print(f"Experiment 2 - Output dir: {output_dir}")
    print(f"Config: {config}")
    correct_dag, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    X_test_original = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    available_orderings = get_ordering_strategies(correct_dag)
    pre_calculated_orders = {}
    for order_strategy in config['ordering_strategies']:
        if order_strategy not in available_orderings:
            raise ValueError(f"Unknown ordering strategy: {order_strategy}. "
                            f"Available: {list(available_orderings.keys())}")
        pre_calculated_orders[order_strategy] = available_orderings[order_strategy]
        print(f"Pre-calculated column order for {order_strategy}: {pre_calculated_orders[order_strategy]}")
    if resume:
        results_so_far, start_train_idx, start_rep = get_checkpoint_info(output_dir)
    else:
        results_so_far, start_train_idx, start_rep = [], 0, 0
    total_iterations = len(config['train_sizes']) * len(config['ordering_strategies']) * config['n_repetitions']
    completed = len(results_so_far)
    print(f"Total iterations: {total_iterations}, Already completed: {completed}")
    hash_check_dict = {}
    try:
        for train_idx, train_size in enumerate(config['train_sizes'][start_train_idx:], start_train_idx):
            for rep in range(start_rep if train_idx == start_train_idx else 0, config['n_repetitions']):
                for order_strategy in config['ordering_strategies']:
                    result = run_single_configuration(
                        train_size, order_strategy, rep, config, X_test_original,
                        correct_dag, col_names, categorical_cols, pre_calculated_orders[order_strategy],
                        data_samples_dir=data_samples_dir if SAVE_DATA_SAMPLES else None,
                        hash_check_dict=hash_check_dict
                    )
                    results_so_far.append(result)
                    df_current = pd.DataFrame(results_so_far)
                    df_current.to_csv(output_dir / "raw_results.csv", index=False)
                    save_checkpoint(results_so_far, train_idx, rep + 1, output_dir)
                    completed += 1
                    print(f"    Progress: {completed}/{total_iterations} ({100*completed/total_iterations:.1f}%)")
                    print(f"    Results saved to: {output_dir}/raw_results.csv")
            start_rep = 0
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Progress saved!")
        return pd.DataFrame(results_so_far)
    print("\nExperiment completed!")
    cleanup_checkpoint(output_dir)
    df_results = pd.DataFrame(results_so_far)
    # Standardize column order for output
    preferred_order = [
        'train_size', 'repetition', 'seed', 'categorical', 'column_order_strategy', 'column_order', 'graph_type', 'graph_structure'
    ]
    metric_cols = [col for col in df_results.columns if col not in preferred_order]
    ordered_cols = [col for col in preferred_order if col in df_results.columns] + metric_cols
    df_results = df_results[ordered_cols]
    df_results.to_csv(output_dir / "raw_results_final.csv", index=False)
    print(f"Results saved to: {output_dir}")
    print(f"Total results: {len(df_results)}")
    return df_results


def main():
    """Main CLI interface for Experiment 2."""
    parser = argparse.ArgumentParser(description='Run Experiment 2: Column ordering effects')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    args = parser.parse_args()
    
    # Show SCM info (only for reference, not used in experiment)
    dag, col_names, _ = get_dag_and_config(False)
    print("=" * 60)
    print("EXPERIMENT 2: Column Ordering Effects on Vanilla TabPFN")
    print("=" * 60)
    print("\nResearch Question:")
    print("Does column ordering affect synthetic data quality when TabPFN")
    print("uses its implicit autoregressive mechanism (no DAG provided)?")
    print("\nSCM structure (for ordering strategies reference):")
    print_dag_info(dag, col_names)
    print()
    
    # Use centralized config
    print("Running FULL experiment...")
    config = DEFAULT_CONFIG.copy()
    output_dir = args.output or "experiment_2_results"
    
    # Calculate total configurations
    total_configs = (len(config['train_sizes']) * 
                    len(config['ordering_strategies']) * 
                    config['n_repetitions'])
    
    print(f"\nExperiment Configuration:")
    print(f"  Training sizes: {config['train_sizes']}")
    print(f"  Ordering strategies: {config['ordering_strategies']}")
    print(f"  Repetitions: {config['n_repetitions']}")
    print(f"  Total configurations: {total_configs}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Output: {output_dir}")
    
    # Run experiment
    results = run_experiment_2(
        config=config,
        output_dir=output_dir,
        resume=not args.no_resume
    )
    
    # Print summary
    if results is not None and len(results) > 0:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Get actual metric columns from results
        metric_columns = [col for col in results.columns if col not in ['train_size', 'repetition', 'categorical', 'seed', 'column_order_strategy', 'column_order', 'graph_type', 'graph_structure']]
        
        # Best and worst orderings per metric
        for metric in metric_columns:
            print(f"\n{metric.upper()}:")
            mean_by_order = results.groupby('column_order_strategy')[metric].mean()
            best_order = mean_by_order.idxmin()
            worst_order = mean_by_order.idxmax()
            
            print(f"  Best ordering: {best_order} ({mean_by_order[best_order]:.4f})")
            print(f"  Worst ordering: {worst_order} ({mean_by_order[worst_order]:.4f})")
            print(f"  Difference: {mean_by_order[worst_order] - mean_by_order[best_order]:.4f}")


if __name__ == "__main__":
    main()