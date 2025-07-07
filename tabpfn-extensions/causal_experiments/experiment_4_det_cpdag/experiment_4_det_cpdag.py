"""
Experiment 4: Causal Knowledge Level Impact on TabPFN Performance.

This experiment tests how different levels of causal knowledge affect TabPFN's
synthetic data generation. We create a CPDAG from the true DAG with controlled
ambiguity, generate all possible DAGs from this CPDAG, and test TabPFN
with DAGs of increasing complexity/completeness.

The CPDAG should be provided as input (e.g., from external causal discovery).
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
import hashlib
from collections import OrderedDict

# Add the causal_experiments directory to the path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# TabPFN imports
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Local imports
from utils.scm_data import generate_scm_data, get_dag_and_config
from utils.metrics import FaithfulDataEvaluator
from utils.dag_utils import cpdag_to_dags, get_ordering_strategies, dag_belongs_to_cpdag, get_graph_edge_counts
from utils.checkpoint_utils import save_checkpoint, get_checkpoint_info, cleanup_checkpoint
from utils.experiment_utils import generate_synthetic_data_quiet, reorder_data_and_columns

SAVE_DATA_SAMPLES = True  # Set to True to save data_samples for debugging

# Centralized default config
DEFAULT_CONFIG = {
    'train_sizes': [20, 50, 100, 200, 500],
    'n_repetitions': 10,
    'test_size': 2000,
    'n_permutations': 3,
    'metrics': ['mean_corr_difference', 'max_corr_difference', 'propensity_metrics', 'k_marginal_tvd'],
    'include_categorical': False,
    'n_estimators': 3,
    'random_seed_base': 42,
    'sample_dags': False,  # Whether to sample DAGs or test all
    'max_dags_to_test': 5,  # Max DAGs to test when sampling
    'cases': ['vanilla', 'dag', 'cpdag'],  # Algorithms to run
}

# Preferred order for result columns
PREFERRED_ORDER = [
    'algorithm', 'graph_type', 'graph_structure', 'dag_edges_directed', 'dag_edges_undirected', 'dag_nodes', 'train_size', 'repetition', 'seed', 'categorical', 'column_order_strategy', 'column_order'
]

# Helper to build result row in correct order
def build_result_row(base_info, metrics, preferred_order, metric_cols):
    row = OrderedDict()
    for k in preferred_order:
        row[k] = base_info.get(k, '')
    for k in metric_cols:
        row[k] = metrics.get(k, '')
    return row

def hash_array(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()

def categorize_dags_by_complexity(dags, max_dags_to_test=5):
    """
    Categorize DAGs by their complexity (number of edges) and select a subset for testing.
    
    Args:
        dags: List of DAG dictionaries
        max_dags_to_test: Maximum number of DAGs to test (including vanilla).
                         If None, test all DAGs.
        
    Returns:
        Dictionary with categories: {category_name: dag}
    """
    if not dags:
        return {'vanilla': None}
    
    # Calculate edge counts for all DAGs
    edge_counts = []
    for dag in dags:
        edge_count = sum(len(parents) for parents in dag.values())
        edge_counts.append((edge_count, dag))
    
    # Sort by edge count
    edge_counts.sort(key=lambda x: x[0])
    
    # Create categories
    categories = {'vanilla': None}  # Always include vanilla case
    
    n_dags = len(edge_counts)
    if n_dags == 0:
        return categories
    
    # If max_dags_to_test is None, include all DAGs
    if max_dags_to_test is None:
        for i, (edge_count, dag) in enumerate(edge_counts):
            categories[f'dag_{i+1}_{edge_count}_edges'] = dag
        return categories
    
    # If we have few DAGs, include all
    if n_dags <= max_dags_to_test - 1:  # -1 for vanilla
        for i, (edge_count, dag) in enumerate(edge_counts):
            categories[f'dag_{i+1}_{edge_count}_edges'] = dag
    else:
        # Sample DAGs across the complexity spectrum
        # Always include the simplest and most complex
        categories[f'dag_min_{edge_counts[0][0]}_edges'] = edge_counts[0][1]
        categories[f'dag_max_{edge_counts[-1][0]}_edges'] = edge_counts[-1][1]
        
        # Add intermediate DAGs if we have space
        remaining_slots = max_dags_to_test - 3  # vanilla, min, max
        if remaining_slots > 0 and n_dags > 2:
            # Sample from the middle
            step = max(1, n_dags // (remaining_slots + 1))
            for i in range(1, min(remaining_slots + 1, n_dags - 1)):
                idx = i * step
                if idx < n_dags - 1:  # Don't include the last one (already included as max)
                    edge_count, dag = edge_counts[idx]
                    categories[f'dag_mid_{i}_{edge_count}_edges'] = dag
    
    return categories

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

# Pipeline: With DAG (no reordering)

def run_with_dag_type(algorithm, graph_type, dag, train_size, repetition, config, X_test, col_names, categorical_cols, data_samples_dir=None, hash_check_dict=None, vanilla_column_order=None):
    seed = config['random_seed_base'] + repetition
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    X_train = generate_scm_data(train_size, seed, config['include_categorical'])
    train_hash = hash_array(X_train)
    test_hash = hash_array(X_test)
    if hash_check_dict is not None:
        key = (algorithm, graph_type, train_size, repetition)
        if key in hash_check_dict:
            prev_train_hash, prev_test_hash = hash_check_dict[key]
            if prev_train_hash != train_hash or prev_test_hash != test_hash:
                raise RuntimeError(f"[HASH ERROR] Train/Test data hash mismatch for algorithm={algorithm}, graph_type={graph_type}, train_size={train_size}, repetition={repetition}!\nPrev train hash: {prev_train_hash}\nCurrent train hash: {train_hash}\nPrev test hash: {prev_test_hash}\nCurrent test hash: {test_hash}")
        else:
            hash_check_dict[key] = (train_hash, test_hash)
    if algorithm == 'vanilla':
        return run_vanilla(X_train, X_test, col_names, categorical_cols, config, seed, train_size, repetition, algorithm, vanilla_column_order=vanilla_column_order, data_samples_dir=data_samples_dir)
    elif algorithm == 'cpdag':
        return run_with_cpdag(X_train, X_test, dag, col_names, categorical_cols, config, seed, train_size, repetition, algorithm)
    elif algorithm == 'dag':
        return run_with_dag(X_train, X_test, dag, col_names, categorical_cols, config, seed, train_size, repetition, algorithm, graph_type)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

# Pipeline: No DAG (with reordering)

def run_vanilla(X_train, X_test, col_names, categorical_cols, config, seed, train_size, repetition, algorithm, vanilla_column_order=None, data_samples_dir=None):
    if vanilla_column_order is None:
        raise ValueError("vanilla_column_order must be provided to run_vanilla")
    if data_samples_dir is None:
        raise ValueError("data_samples_dir must be provided to run_vanilla")
    X_train_reordered, col_names_reordered, categorical_cols_reordered = reorder_data_and_columns(
        X_train, col_names, categorical_cols, vanilla_column_order
    )
    X_test_reordered, _, _ = reorder_data_and_columns(
        X_test, col_names, categorical_cols, vanilla_column_order
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    if categorical_cols_reordered:
        model.set_categorical_features(categorical_cols_reordered)
    model.fit(torch.from_numpy(X_train_reordered).float())
    X_synth = generate_synthetic_data_quiet(
        model, n_samples=X_test.shape[0], dag=None, n_permutations=config['n_permutations']
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if SAVE_DATA_SAMPLES and data_samples_dir is not None:
        os.makedirs(data_samples_dir, exist_ok=True)
        file_prefix = f"dag_{algorithm}_size{train_size}_rep{repetition}"
        pd.DataFrame(X_train_reordered, columns=col_names_reordered).to_csv(os.path.join(data_samples_dir, f"{file_prefix}_train.csv"), index=False)
        pd.DataFrame(X_test_reordered, columns=col_names_reordered).to_csv(os.path.join(data_samples_dir, f"{file_prefix}_test.csv"), index=False)
        pd.DataFrame(X_synth, columns=col_names_reordered).to_csv(os.path.join(data_samples_dir, f"{file_prefix}_synth.csv"), index=False)
    metrics = evaluate_metrics(X_test_reordered, X_synth, col_names_reordered, categorical_cols_reordered)
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': 'original',
        'column_order': str(vanilla_column_order),
        'algorithm': algorithm,
        'graph_type': None,
        'graph_structure': '',
        'dag_edges_directed': 0,
        'dag_edges_undirected': 0,
        'dag_nodes': 0,
        'dag_structure': 'None',
    }
    # Flatten metrics
    flat_metrics = {}
    for metric in config['metrics']:
        value = metrics.get(metric)
        if isinstance(value, dict):
            for submetric, subvalue in value.items():
                flat_metrics[f'{metric}_{submetric}'] = subvalue
        else:
            flat_metrics[metric] = value
    metric_cols = list(flat_metrics.keys())
    result_row = build_result_row(base_info, flat_metrics, PREFERRED_ORDER, metric_cols)
    return result_row

def run_with_dag(X_train, X_test, dag, col_names, categorical_cols, config, seed, train_size, repetition, algorithm, graph_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    if categorical_cols:
        model.set_categorical_features(categorical_cols)
    model.fit(torch.from_numpy(X_train).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], dag=dag, n_permutations=config['n_permutations']
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = evaluate_metrics(X_test, X_synth, col_names, categorical_cols)
    dag_dict = dag
    edge_counts = get_graph_edge_counts(dag_dict)
    dag_nodes = len(dag_dict)
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': '',
        'column_order': '',
        'algorithm': algorithm,
        'graph_type': graph_type,
        'graph_structure': str(dag_dict),
        'dag_edges_directed': edge_counts['directed'],
        'dag_edges_undirected': edge_counts['undirected'],
        'dag_nodes': dag_nodes,
        'dag_structure': str(dag_dict),
    }
    # Flatten metrics
    flat_metrics = {}
    for metric in config['metrics']:
        value = metrics.get(metric)
        if isinstance(value, dict):
            for submetric, subvalue in value.items():
                flat_metrics[f'{metric}_{submetric}'] = subvalue if subvalue is not None else ''
        else:
            flat_metrics[metric] = value if value is not None else ''
    metric_cols = list(flat_metrics.keys())
    result_row = build_result_row(base_info, flat_metrics, PREFERRED_ORDER, metric_cols)
    return result_row

def run_with_cpdag(X_train, X_test, cpdag, col_names, categorical_cols, config, seed, train_size, repetition, algorithm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = TabPFNClassifier(n_estimators=config['n_estimators'], device=device)
    reg = TabPFNRegressor(n_estimators=config['n_estimators'], device=device)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    if categorical_cols:
        model.set_categorical_features(categorical_cols)
    model.fit(torch.from_numpy(X_train).float())
    X_synth = generate_synthetic_data_quiet(
        model, config['test_size'], cpdag=cpdag, n_permutations=config['n_permutations']
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = evaluate_metrics(X_test, X_synth, col_names, categorical_cols)
    # Convert CPDAG to dict if needed and count edges
    if isinstance(cpdag, np.ndarray):
        cpdag_dict = model._parse_cpdag_adjacency_matrix(cpdag)
    else:
        cpdag_dict = cpdag
    edge_counts = get_graph_edge_counts(cpdag_dict)
    dag_nodes = len(cpdag_dict)
    base_info = {
        'train_size': train_size,
        'repetition': repetition,
        'seed': seed,
        'categorical': config['include_categorical'],
        'column_order_strategy': '',
        'column_order': '',
        'algorithm': algorithm,
        'graph_type': 'cpdag',
        'graph_structure': str(cpdag_dict),
        'dag_edges_directed': edge_counts['directed'],
        'dag_edges_undirected': edge_counts['undirected'],
        'dag_nodes': dag_nodes,
        'dag_structure': str(cpdag_dict),
    }
    # Flatten metrics
    flat_metrics = {}
    for metric in config['metrics']:
        value = metrics.get(metric)
        if isinstance(value, dict):
            for submetric, subvalue in value.items():
                flat_metrics[f'{metric}_{submetric}'] = subvalue if subvalue is not None else ''
        else:
            flat_metrics[metric] = value if value is not None else ''
    metric_cols = list(flat_metrics.keys())
    result_row = build_result_row(base_info, flat_metrics, PREFERRED_ORDER, metric_cols)
    return result_row

# Main configuration orchestrator

def run_single_configuration(train_size, dag_level, repetition, config, 
                           X_test, dag_categories, col_names, categorical_cols, vanilla_column_order, data_samples_dir=None, hash_check_dict=None):
    print(f"    DAG level: {dag_level}, Rep: {repetition+1}/{config['n_repetitions']}")
    seed = config['random_seed_base'] + repetition
    X_train = generate_scm_data(
        n_samples=train_size,
        random_state=seed,
        include_categorical=config['include_categorical']
    )
    train_hash = hash_array(X_train)
    test_hash = hash_array(X_test)
    if hash_check_dict is not None:
        key = (train_size, repetition)
        if key in hash_check_dict:
            prev_train_hash, prev_test_hash = hash_check_dict[key]
            if prev_train_hash != train_hash or prev_test_hash != test_hash:
                raise RuntimeError(f"[HASH ERROR] Train/Test data hash mismatch for train_size={train_size}, repetition={repetition}!\nPrev train hash: {prev_train_hash}\nCurrent train hash: {train_hash}\nPrev test hash: {prev_test_hash}\nCurrent test hash: {test_hash}")
        else:
            hash_check_dict[key] = (train_hash, test_hash)
    dag_to_use = dag_categories[dag_level]
    if dag_level == 'vanilla':
        print(f"    Using pre-calculated column order: original = {vanilla_column_order}")
        return run_vanilla(X_train, X_test, col_names, categorical_cols, config, seed, train_size, repetition, dag_level, vanilla_column_order=vanilla_column_order, data_samples_dir=data_samples_dir)
    else:
        return run_with_dag_type(dag_level, dag_to_use, train_size, repetition, config, X_test, col_names, categorical_cols, data_samples_dir=data_samples_dir, hash_check_dict=hash_check_dict, vanilla_column_order=vanilla_column_order)

def run_experiment_4(cpdag, config=None, output_dir="experiment_4_results", resume=True):
    """
    Main experiment function for testing causal knowledge levels.
    """
    # Use centralized config and update with any overrides
    base_config = DEFAULT_CONFIG.copy()
    if config is not None:
        base_config.update(config)
    config = base_config
    cases = config.get('cases', ['vanilla', 'dag'])
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir) / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Create data_samples directory
    data_samples_dir = Path(script_dir) / 'data_samples'
    data_samples_dir.mkdir(exist_ok=True)
    
    print(f"Experiment 4 - Output dir: {output_dir}")
    print(f"Config: {config}")
    
    # Setup
    _, col_names, categorical_cols = get_dag_and_config(config['include_categorical'])
    X_test = generate_scm_data(config['test_size'], 123, config['include_categorical'])
    
    # Generate all possible DAGs from CPDAG
    print("Generating DAGs from CPDAG...")
    all_dags = cpdag_to_dags(cpdag)
    print(f"Generated {len(all_dags)} DAGs from CPDAG")

    # Categorize DAGs by complexity
    max_dags = config['max_dags_to_test'] if config['sample_dags'] else None
    dag_categories = categorize_dags_by_complexity(all_dags, max_dags)

    # --- Inserted logic for correct DAG handling ---
    # Get the true DAG and column names
    true_dag, col_names, _ = get_dag_and_config(config['include_categorical'])
    cpdag_adj = cpdag
    # Normalize function for robust comparison
    def normalize_dag_dict(dag_dict):
        return {k: sorted(v) for k, v in dag_dict.items()}
    # Check if the true DAG is among the discovered DAGs
    true_dag_found = False
    key_to_rename = None
    for key, dag in dag_categories.items():
        if dag is not None and normalize_dag_dict(dag) == normalize_dag_dict(true_dag):
            true_dag_found = True
            key_to_rename = key
            break
    if true_dag_found and key_to_rename is not None:
        # Rename the key to indicate it is the correct DAG (use _correct, no space)
        dag_categories[f"{key_to_rename}_correct"] = dag_categories.pop(key_to_rename)
        print(f"True DAG found among discovered DAGs. Renamed '{key_to_rename}' to '{key_to_rename}_correct'.")
    else:
        # Add the true DAG as an extra pipeline
        dag_categories["dag_CORRECT"] = true_dag
        print("True DAG was NOT found among discovered DAGs. Added as 'dag_CORRECT'.")
    
    print(f"\nAlgorithms/Graphs to test:")
    print("-" * 40)
    for name, dag in dag_categories.items():
        if name.lower().startswith('vanilla'):
            print(f"  {name}: Algorithm (no graph provided, TabPFN vanilla)")
        elif 'cpdag' in name.lower():
            print(f"  {name}: Algorithm (CPDAG provided, equivalence class of DAGs)")
        elif 'correct' in name.lower():
            print(f"  {name}: Algorithm (true DAG provided)")
        else:
            edge_count = sum(len(parents) for parents in dag.values()) if dag is not None else 0
            print(f"  {name}: Graph ({edge_count} edges)")
    
    # Pre-calculate column order for vanilla case (ONCE!)
    # Use the first available DAG for getting ordering strategies
    first_dag = next((dag for dag in dag_categories.values() if dag is not None), None)
    if first_dag is None:
        # If no DAGs available, raise an error instead of using hardcoded fallback
        raise ValueError("No DAGs available for determining column ordering. Check CPDAG generation.")
    available_orderings = get_ordering_strategies(first_dag)
    vanilla_column_order = available_orderings['original']
    print(f"Pre-calculated column order for vanilla case: original = {vanilla_column_order}")
    
    # Check for checkpoint
    if resume:
        results_so_far, start_train_idx, start_rep = get_checkpoint_info(output_dir, "experiment_4_checkpoint.pkl")
    else:
        results_so_far, start_train_idx, start_rep = [], 0, 0
    
    # Run experiment
    total_iterations = len(config['train_sizes']) * len(dag_categories) * config['n_repetitions']
    completed = len(results_so_far)
    
    print(f"Total iterations: {total_iterations}, Already completed: {completed}")
    
    try:
        hash_check_dict = {}
        config_idx = 0
        # Iterate in the order of 'cases', not dag_categories
        for case in cases:
            if case == 'vanilla':
                if 'vanilla' in dag_categories:
                    algorithm = 'vanilla'
                    graph_type = None
                    dag = dag_categories['vanilla']
                    print(f"\n=== Running algorithm: {algorithm} ===")
                    for train_idx, train_size in enumerate(config['train_sizes']):
                        for rep in range(config['n_repetitions']):
                            if config_idx < completed:
                                config_idx += 1
                                continue
                            row = run_with_dag_type(
                                algorithm, graph_type, dag, train_size, rep, config, X_test, col_names, categorical_cols, data_samples_dir=data_samples_dir, hash_check_dict=hash_check_dict, vanilla_column_order=vanilla_column_order
                            )
                            results_so_far.append(row)
                            df_current = pd.DataFrame(results_so_far)
                            df_current.to_csv(output_dir / "raw_results.csv", index=False, na_rep='')
                            save_checkpoint(results_so_far, train_idx, rep + 1, output_dir, "experiment_4_checkpoint.pkl")
                            completed += 1
                            config_idx += 1
                            print(f"    Progress ({algorithm}): {completed}/{total_iterations} ({100*completed/total_iterations:.1f}%)")
                            print(f"    Results saved to: {output_dir / 'raw_results.csv'}")
            elif case == 'dag':
                for dag_level, dag in dag_categories.items():
                    if dag_level.lower().startswith('vanilla'):
                        continue
                    algorithm = 'dag'
                    graph_type = dag_level
                    print(f"\n=== Running algorithm: {algorithm} (graph: {graph_type}) ===")
                    for train_idx, train_size in enumerate(config['train_sizes']):
                        for rep in range(config['n_repetitions']):
                            if config_idx < completed:
                                config_idx += 1
                                continue
                            row = run_with_dag_type(
                                algorithm, graph_type, dag, train_size, rep, config, X_test, col_names, categorical_cols, data_samples_dir=data_samples_dir, hash_check_dict=hash_check_dict, vanilla_column_order=vanilla_column_order
                            )
                            results_so_far.append(row)
                            df_current = pd.DataFrame(results_so_far)
                            df_current.to_csv(output_dir / "raw_results.csv", index=False, na_rep='')
                            save_checkpoint(results_so_far, train_idx, rep + 1, output_dir, "experiment_4_checkpoint.pkl")
                            completed += 1
                            config_idx += 1
                            print(f"    Progress ({algorithm}/{graph_type}): {completed}/{total_iterations} ({100*completed/total_iterations:.1f}%)")
                            print(f"    Results saved to: {output_dir / 'raw_results.csv'}")
            elif case == 'cpdag':
                algorithm = 'cpdag'
                graph_type = 'cpdag'
                dag = cpdag
                print(f"\n=== Running algorithm: {algorithm} (graph: {graph_type}) ===")
                for train_idx, train_size in enumerate(config['train_sizes']):
                    for rep in range(config['n_repetitions']):
                        if config_idx < completed:
                            config_idx += 1
                            continue
                        row = run_with_dag_type(
                            algorithm, graph_type, dag, train_size, rep, config, X_test, col_names, categorical_cols, data_samples_dir=data_samples_dir, hash_check_dict=hash_check_dict, vanilla_column_order=vanilla_column_order
                        )
                        results_so_far.append(row)
                        df_current = pd.DataFrame(results_so_far)
                        df_current.to_csv(output_dir / "raw_results.csv", index=False, na_rep='')
                        save_checkpoint(results_so_far, train_idx, rep + 1, output_dir, "experiment_4_checkpoint.pkl")
                        completed += 1
                        config_idx += 1
                        print(f"    Progress ({algorithm}): {completed}/{total_iterations} ({100*completed/total_iterations:.1f}%)")
                        print(f"    Results saved to: {output_dir / 'raw_results.csv'}")
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Progress saved!")
        return pd.DataFrame(results_so_far)
    
    # The true DAG testing is already included in the main loop above
    # If true_dag_found is True, it was renamed and tested in dag_categories
    # If true_dag_found is False, it was added as "dag_CORRECT" and tested in dag_categories
    # Therefore, no additional true_dag testing is needed here
    
    # Experiment completed
    print("\nExperiment completed!")
    
    # Clean up checkpoint
    cleanup_checkpoint(output_dir, "experiment_4_checkpoint.pkl")
    
    # Final results
    df_results = pd.DataFrame(results_so_far)
    # Standardize column order for output
    preferred_order = [
        'algorithm', 'graph_type', 'graph_structure', 'dag_edges_directed', 'dag_edges_undirected', 'dag_nodes', 'train_size', 'repetition', 'seed', 'categorical', 'column_order_strategy', 'column_order'
    ]
    metric_cols = [col for col in df_results.columns if col not in preferred_order]
    ordered_cols = [col for col in preferred_order if col in df_results.columns] + metric_cols
    df_results = df_results[ordered_cols]
    df_results.to_csv(output_dir / "experiment_4_results.csv", index=False, na_rep='')
    
    print(f"Results saved to: {output_dir}")
    print(f"Total results: {len(df_results)}")
    
    return df_results 


if __name__ == "__main__":
    import argparse
    from run_pc_discovery import run_pc_discovery_on_dataset
    from utils.scm_data import generate_scm_data, get_dag_and_config

    parser = argparse.ArgumentParser(
        description='Run Experiment 4: Causal Knowledge Level Impact',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--no-resume', action='store_true',
                       help='Start a fresh run (ignores any existing checkpoint).')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (a default name will be generated if not specified).')
    parser.add_argument('--include-categorical', action='store_true',
                        help='Include categorical variables in the SCM (default: only continuous variables).')
    parser.add_argument('--sample-dags', action='store_true',
                        help='Sample DAGs by complexity (default: test all possible DAGs).')

    args = parser.parse_args()

    print("Starting Experiment 4: Causal Knowledge Level Impact")
    print(f"Data type: {'Mixed (continuous + categorical)' if args.include_categorical else 'Continuous only'}")
    print(f"DAG sampling: {'Enabled (max 5 DAGs)' if args.sample_dags else 'Disabled (all DAGs)'}")
    print("=" * 60)

    config = DEFAULT_CONFIG.copy()
    config['include_categorical'] = args.include_categorical
    config['sample_dags'] = args.sample_dags
    config['max_dags_to_test'] = 5 if args.sample_dags else None
    output_dir = args.output or f"experiment_4_results_{'mixed' if args.include_categorical else 'continuous'}"

    # Discovery step
    true_dag, col_names, categorical_cols = get_dag_and_config(
        include_categorical=args.include_categorical
    )
    n_discovery_samples = 2000
    print(f"Generating {n_discovery_samples} samples for PC discovery...")
    X_discovery = generate_scm_data(
        n_samples=n_discovery_samples,
        random_state=config['random_seed_base'],
        include_categorical=args.include_categorical
    )
    print("Discovering CPDAG from data using PC algorithm...")
    cpdag = run_pc_discovery_on_dataset(
        dataset_name="mixed" if args.include_categorical else "continuous",
        data=X_discovery,
        true_dag=true_dag,
        task_type="classification" if "target" in col_names else "unsupervised",
        target_column="target" if "target" in col_names else None,
        verbose=False,
        output_dir=None,
    )
    print(f"CPDAG discovered successfully.")

    # Run experiment
    run_experiment_4(
        cpdag=cpdag,
        config=config,
        output_dir=output_dir,
        resume=not args.no_resume
    )

    print("\n" + "=" * 50)
    print("All experiments finished.")
    print(f"Results saved in: {output_dir}")
    print("=" * 50) 